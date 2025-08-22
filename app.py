from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import streamlit as st
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Get Groq API key
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise ValueError("GROQ_API_KEY is missing from the .env file")

# LangSmith tracking (optional)
langsmith_api_key = os.getenv("LANGCHAIN_API_KEY")
if langsmith_api_key:
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = langsmith_api_key
    os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGSMITH_PROJECT", "resume-assistant")
    os.environ["LANGCHAIN_ENDPOINT"] = os.getenv("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")

# Prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are Resume Assistant. Please respond to user queries."),
    ("human", "Query: {query}")
])

# Streamlit UI
st.title(" Resume Assistant (Groq + LangChain + LLaMA3)")
st.write("Ask resume-related or job-search questions below.")

# User input
query = st.text_input("Enter your question")

# LLM + Output parser
llm = ChatGroq(groq_api_key=groq_api_key, model="llama3-8b-8192")
parser = StrOutputParser()
chain = prompt | llm | parser

# Handle query
if query:
    with st.spinner("Thinking..."):
        response = chain.invoke({"query": query})
        st.write("### Answer")
        st.write(response)
