import streamlit as st
import os
from dotenv import load_dotenv

# Import all necessary LLM and agent components
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain.agents.agent_types import AgentType
from business_insights_generator.dataset import load_sales_data, load_review_data

# LangGraph Imports
from typing import TypedDict, Annotated, Sequence
import operator
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.tools import Tool
from langchain_core.pydantic_v1 import BaseModel, Field

# Streamlit Page Configuration
st.set_page_config(page_title="Analyst AI", layout="wide")
st.title("🤖 Analyst AI")
st.write("I am an AI-powered business analyst. Ask me anything about your sales and customer review data.")

# --- API Key and Data Loading ---
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
google_api_key = os.getenv("GOOGLE_API_KEY")

# Configure Google API if key exists
if google_api_key:
    genai.configure(api_key=google_api_key)

@st.cache_data
def get_data(data_type):
    if data_type == 'sales': return load_sales_data()
    if data_type == 'reviews': return load_review_data()

sales_df = get_data('sales')
reviews_df = get_data('reviews')

# --- Tool/Agent Creation with Fallback Logic ---
@st.cache_resource
def get_tools_and_llm():
    """
    Initializes the LLM with fallback logic and creates the specialist agent tools.
    Returns both the list of tools and the successfully initialized LLM instance.
    """
    primary_model = "llama-3.3-70b-versatile"
    fallback_model = "llama-3.1-8b-instant"
    
    try:
        llm = ChatGroq(temperature=0, model_name=primary_model, groq_api_key=groq_api_key)
        st.sidebar.success(f"Using primary model: {primary_model}")
    except Exception as e:
        st.sidebar.warning(f"Primary model ({primary_model}) failed: {e}. Switching to fallback.")
        try:
            llm = ChatGroq(temperature=0, model_name=fallback_model, groq_api_key=groq_api_key)
            st.sidebar.success(f"Using fallback model: {fallback_model}")
        except Exception as e2:
            st.sidebar.error(f"All Groq models failed: {e2}. Using Google Gemini as last resort.")
            if google_api_key:
                llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0)
                st.sidebar.success("Using final backup model: Google Gemini")
            else:
                st.sidebar.error("Google API key not found. Cannot use backup model.")
                return None, None

    sales_agent_executor = create_pandas_dataframe_agent(llm, sales_df, agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION, allow_dangerous_code=True, handle_parsing_errors=True, agent_executor_kwargs={'handle_parsing_errors': True})
    review_agent_executor = create_pandas_dataframe_agent(llm, reviews_df, agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION, allow_dangerous_code=True, handle_parsing_errors=True, agent_executor_kwargs={'handle_parsing_errors': True})

    def sales_tool_wrapper(query: str) -> str:
        response = sales_agent_executor.invoke({"input": query})
        return response.get('output', 'An error occurred in the sales agent.')

    def review_tool_wrapper(query: str) -> str:
        response = review_agent_executor.invoke({"input": query})
        return response.get('output', 'An error occurred in the review agent.')

    class ToolInput(BaseModel):
        query: str = Field(description="The user's question for the specialist agent.")

    tools = [
        Tool(name="SalesAgent", func=sales_tool_wrapper, description="Use for sales, revenue, quantities.", args_schema=ToolInput),
        Tool(name="ReviewAgent", func=review_tool_wrapper, description="Use for customer feedback, sentiment, ratings.", args_schema=ToolInput),
    ]
    return tools, llm

# --- Main Logic to Initialize and Run Graph ---
if sales_df is not None and reviews_df is not None:
    tools, llm = get_tools_and_llm()

    if tools and llm:
        class AgentState(TypedDict):
            messages: Annotated[Sequence[BaseMessage], operator.add]

        @st.cache_resource
        def get_graph(_llm, _tools):
            model_with_tools = _llm.bind_tools(_tools)

            def agent_node(state, config):
                return {"messages": [model_with_tools.invoke(state["messages"], config)]}

            tool_node = ToolNode(_tools)

            def router(state):
                last_message = state["messages"][-1]
                if getattr(last_message, "tool_calls", None):
                    return "tools"
                return "end"

            workflow = StateGraph(AgentState)
            workflow.add_node("agent", agent_node)
            workflow.add_node("tools", tool_node)
            workflow.set_entry_point("agent")
            workflow.add_conditional_edges("agent", router, {"tools": "tools", "end": END})
            workflow.add_edge("tools", "agent")
            return workflow.compile()

        graph = get_graph(llm, tools)

        # --- UI Logic ---
        def handle_query(user_question):
            st.session_state.messages.append({"role": "user", "content": user_question})
            with st.chat_message("user"):
                st.markdown(user_question)

            with st.chat_message("assistant", avatar="🤖"):
                with st.spinner("Analyst AI is digging into the data..."):
                    try:
                        final_state = graph.invoke(
                            {"messages": [HumanMessage(content=user_question)]},
                            config={"recursion_limit": 50}
                        )
                        final_answer = final_state['messages'][-1].content
                        st.markdown(final_answer)
                        st.session_state.messages.append({"role": "assistant", "content": final_answer})
                    except Exception as e:
                        error_message = f"An error occurred: {e}"
                        st.error(error_message)
                        st.session_state.messages.append({"role": "assistant", "content": error_message})

        if "messages" not in st.session_state:
            st.session_state.messages = []

        if not st.session_state.messages:
            with st.chat_message("assistant", avatar="🤖"):
                st.write("Hello! How can I help you analyze your business data today?")
            
            st.subheader("Or try one of these suggestions:")
            suggested_questions = [
                "What are the top 3 selling product categories by revenue?",
                "What is the average review rating?",
                "Show me the sales trend over the last 6 months.",
            ]
            
            # Create columns with a maximum of 4 buttons per row
            max_cols = 4
            num_questions = len(suggested_questions)
            for i in range(0, num_questions, max_cols):
                cols = st.columns(max_cols)
                for j in range(max_cols):
                    if i + j < num_questions:
                        question = suggested_questions[i+j]
                        if cols[j].button(question, use_container_width=True):
                            handle_query(question)
                            st.rerun()

        for message in st.session_state.messages:
            with st.chat_message(message["role"], avatar="🤖" if message["role"] == "assistant" else None):
                st.markdown(message["content"])

        if user_question := st.chat_input("Ask me anything about your business data..."):
            handle_query(user_question)