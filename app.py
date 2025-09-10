import streamlit as st
import os
from dotenv import load_dotenv

# --- NEW IMPORTS for Groq and Llama 3 ---
from langchain_groq import ChatGroq

# --- LangChain components for the agent ---
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent

# --- Your custom function to load the data ---
from business_insights_generator.dataset import load_sales_data

# --- Streamlit Page Configuration ---
st.set_page_config(page_title="AI Business Insights Generator", layout="wide")
st.title("🤖 AI-Powered Business Insights Generator")
st.write("Welcome! Ask a question about your business data, and let AI provide you with a clear analysis.")

# --- API Key and Model Configuration ---
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    st.error("Groq API Key not found. Please add GROQ_API_KEY to your .env file.")
    st.stop() # Stop the app if the key is not found

# --- Caching Functions for Performance ---
# @st.cache_data caches the output of the function, so we don't reload the data on every interaction.
@st.cache_data
def get_sales_data():
    try:
        return load_sales_data()
    except FileNotFoundError as e:
        st.error(f"Data Loading Error: {e}")
        return None

# @st.cache_resource caches the created agent, so we don't re-initialize it on every interaction.
@st.cache_resource
def create_agent(df):
    """
    Creates and caches a LangChain agent to interact with the sales DataFrame using Groq.
    """
    # Initialize the Groq LLM with the Llama 3 model
    llm = ChatGroq(
        temperature=0,
        model_name="llama-3.3-70b-versatile",
        groq_api_key=groq_api_key
    )
    
    # A custom error message for robust parsing
    error_message = (
        "I apologize, but I encountered an issue formatting the final answer. "
        "Here is the direct output from my analysis:"
    )

    return create_pandas_dataframe_agent(
        llm,
        df,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        allow_dangerous_code=True,
        handle_parsing_errors=error_message
    )

# --- Main Application Logic ---
sales_df = get_sales_data()

if sales_df is not None:
    # Display a preview of the data to the user
    with st.expander("Preview the first 5 rows of the sales data"):
        st.dataframe(sales_df.head())

    # Create the agent
    agent = create_agent(sales_df)
    
    # User input
    st.subheader("Ask Your Question")
    placeholder_text = "e.g., 'What are our top 3 selling products?' or 'Which region has the lowest sales?'"
    user_question = st.text_input("Enter your question here:", placeholder=placeholder_text)

    if st.button("Generate Insight"):
        if user_question:
            with st.spinner("The AI agent is analyzing the data..."):
                try:
                    # The agent.invoke() call is where the magic happens
                    response = agent.invoke(user_question)
                    
                    st.subheader("💡 AI Analysis:")
                    st.success(response['output'])

                except Exception as e:
                    st.error(f"An error occurred while running the agent: {e}")
        else:
            st.warning("Please enter a question.")