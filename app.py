import streamlit as st
import pandas as pd
import os
from dotenv import load_dotenv

# Import all necessary LLM and agent components
from langchain_groq import ChatGroq
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain.agents.agent_types import AgentType

# LangGraph Imports
from typing import TypedDict, Annotated, Sequence
import operator
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.tools import Tool
from langchain_core.pydantic_v1 import BaseModel, Field

# --- Streamlit Page Configuration ---
st.set_page_config(page_title="Analyst AI", layout="wide")
st.title("🤖 Analyst AI")
st.write("I am an AI-powered business analyst. Upload one or both of your data files (as **CSV only**) to get started.")

# --- NEW: Detailed Upload Guidelines in the Sidebar ---
st.sidebar.markdown("### ⚠️ Upload Guidelines")
st.sidebar.info(
    """
    - **File Format:** Must be a CSV file.
    - **Max File Size:** 100 MB per file.
    - **Data Format:** Data should be in a standard tabular format with a header row.
    
    **For Sales Data:**
    Your CSV **must** contain columns that can be identified as:
    - A **date column** (e.g., `Date`, `Order_Date`, `InvoiceDate`).
    - A **numeric amount column** (e.g., `Amount`, `Price`, `Revenue`, `Total`).
    
    **For Review Data:**
    Your CSV **must** contain these exact column names (case-insensitive):
    - `asins`
    - `reviews.rating`
    - `reviews.text`
    
    *The app will attempt to automatically clean the data, handle missing values, and correct data types based on these required columns.*
    """
)

# --- API Key Configuration ---
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    st.error("Groq API Key not found. Please add GROQ_API_KEY to your .env file.")
    st.stop()

# --- Flexible and Robust Cleaning Functions ---
def clean_sales_data(df: pd.DataFrame) -> pd.DataFrame | None:
    st.sidebar.write("Processing sales data...")
    cleaned_df = df.copy()
    cleaned_df.columns = [str(col).lower() for col in cleaned_df.columns]

    date_aliases = ['date', 'order_date', 'invoicedate']
    amount_aliases = ['amount', 'price', 'unitprice', 'revenue', 'total']
    
    date_col = next((col for col in cleaned_df.columns if col in date_aliases), None)
    amount_col = next((col for col in cleaned_df.columns if col in amount_aliases), None)

    if not date_col or not amount_col:
        st.sidebar.error("Sales CSV is missing a required date or amount/price column.")
        return None

    cleaned_df.rename(columns={date_col: 'date', amount_col: 'amount'}, inplace=True)
    cleaned_df['date'] = pd.to_datetime(cleaned_df['date'], errors='coerce')
    cleaned_df['amount'] = pd.to_numeric(cleaned_df['amount'], errors='coerce')
    cleaned_df.dropna(subset=['date', 'amount'], inplace=True)
    
    st.sidebar.success("Sales data processed.")
    return cleaned_df

def clean_review_data(df: pd.DataFrame) -> pd.DataFrame | None:
    st.sidebar.write("Processing review data...")
    cleaned_df = df.copy()
    cleaned_df.columns = [str(col).lower() for col in cleaned_df.columns]

    required_cols_map = {'asins': 'product_id', 'reviews.rating': 'rating', 'reviews.text': 'review_text'}
    
    if not all(col in cleaned_df.columns for col in required_cols_map.keys()):
        st.sidebar.error("Review CSV must contain 'asins', 'reviews.rating', and 'reviews.text'.")
        return None
        
    cleaned_df = cleaned_df[list(required_cols_map.keys())].copy()
    cleaned_df.rename(columns=required_cols_map, inplace=True)
    cleaned_df.dropna(inplace=True)
    cleaned_df['rating'] = pd.to_numeric(cleaned_df['rating'], errors='coerce').astype('Int64')
    cleaned_df.dropna(subset=['rating'], inplace=True)
    
    st.sidebar.success("Review data processed.")
    return cleaned_df

# --- File Upload and State Management ---
st.sidebar.header("📂 Data Upload")
sales_file = st.sidebar.file_uploader("Upload Sales CSV", type="csv")
reviews_file = st.sidebar.file_uploader("Upload Reviews CSV", type="csv")

# Process files as they are uploaded and store in session state
if sales_file is not None and "sales_df" not in st.session_state:
    st.session_state.sales_df = clean_sales_data(pd.read_csv(sales_file, low_memory=False))
if reviews_file is not None and "reviews_df" not in st.session_state:
    st.session_state.reviews_df = clean_review_data(pd.read_csv(reviews_file, low_memory=False))

# --- Dynamically Build Tools Based on Uploaded Data ---
tools = []
# Use a placeholder for the LLM that will be initialized only if tools are created
llm = None 
if 'sales_df' in st.session_state and st.session_state.sales_df is not None:
    llm = ChatGroq(temperature=0, model_name="llama-3.1-70b-versatile", groq_api_key=groq_api_key)
    sales_agent_executor = create_pandas_dataframe_agent(llm, st.session_state.sales_df, agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION, allow_dangerous_code=True, handle_parsing_errors=True, agent_executor_kwargs={'handle_parsing_errors': True})
    def sales_tool_wrapper(query: str) -> str:
        response = sales_agent_executor.invoke({"input": query})
        return response.get('output', 'Error in sales agent.')
    class ToolInput(BaseModel):
        query: str = Field(description="The user's question for the agent.")
    tools.append(Tool(name="SalesAgent", func=sales_tool_wrapper, description="Use for questions about sales, revenue, quantities, and financial metrics.", args_schema=ToolInput))

if 'reviews_df' in st.session_state and st.session_state.reviews_df is not None:
    if llm is None: # Initialize LLM if not already done
        llm = ChatGroq(temperature=0, model_name="llama-3.3-70b-versatile", groq_api_key=groq_api_key)
    review_agent_executor = create_pandas_dataframe_agent(llm, st.session_state.reviews_df, agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION, allow_dangerous_code=True, handle_parsing_errors=True, agent_executor_kwargs={'handle_parsing_errors': True})
    def review_tool_wrapper(query: str) -> str:
        response = review_agent_executor.invoke({"input": query})
        return response.get('output', 'Error in review agent.')
    class ToolInput(BaseModel):
        query: str = Field(description="The user's question for the agent.")
    tools.append(Tool(name="ReviewAgent", func=review_tool_wrapper, description="Use for questions about customer feedback, sentiment, and product ratings.", args_schema=ToolInput))

# --- Main Application Logic ---
if tools: # Only show the chat interface if at least one tool is ready
    model_with_tools = llm.bind_tools(tools)
    
    class AgentState(TypedDict):
        messages: Annotated[Sequence[BaseMessage], operator.add]

    @st.cache_resource
    def get_graph():
        def agent_node(state, config): return {"messages": [model_with_tools.invoke(state["messages"], config)]}
        tool_node = ToolNode(tools)
        def router(state):
            if getattr(state["messages"][-1], "tool_calls", None): return "tools"
            return "end"
        workflow = StateGraph(AgentState)
        workflow.add_node("agent", agent_node)
        workflow.add_node("tools", tool_node)
        workflow.set_entry_point("agent")
        workflow.add_conditional_edges("agent", router, {"tools": "tools", "end": END})
        workflow.add_edge("tools", "agent")
        return workflow.compile()

    graph = get_graph()

    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"], avatar="🤖" if message["role"] == "assistant" else None):
            st.markdown(message["content"])

    if user_question := st.chat_input("Ask me anything about the uploaded data..."):
        st.session_state.messages.append({"role": "user", "content": user_question})
        with st.chat_message("user"):
            st.markdown(user_question)

        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("Analyst AI is digging into the data..."):
                try:
                    final_state = graph.invoke({"messages": [HumanMessage(content=user_question)]}, config={"recursion_limit": 50})
                    final_answer = final_state['messages'][-1].content
                    st.markdown(final_answer)
                    st.session_state.messages.append({"role": "assistant", "content": final_answer})
                except Exception as e:
                    error_message = f"An error occurred: {e}"
                    st.error(error_message)
                    st.session_state.messages.append({"role": "assistant", "content": error_message})
else:
    st.info("Please upload at least one CSV file in the sidebar to begin the analysis.")