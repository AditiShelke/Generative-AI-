from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import streamlit as st
from dotenv import load_dotenv
import os

# Load environment variables from .env
load_dotenv()

# Get OpenAI API key
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key: 
    raise ValueError("OPENAI_API_KEY is missing from the .env file")

# Optional: LangSmith tracking
langsmith_api_key = os.getenv("LANGCHAIN_API_KEY")
langsmith_project = os.getenv("LANGSMITH_PROJECT")
langsmith_endpoint = os.getenv("LANGSMITH_ENDPOINT")

if langsmith_api_key:
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = langsmith_api_key
    if langsmith_project:
        os.environ["LANGCHAIN_PROJECT"] = langsmith_project
    if langsmith_endpoint:
        os.environ["LANGCHAIN_ENDPOINT"] = langsmith_endpoint

# Prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are Resume Assistant. Please respond to user queries."),
    ("human", "Query: {query}")
])

# Streamlit UI
st.title(" Resume Assistant with LangChain + OpenAI")
st.write("Ask questions about your resume or job applications!")
query = st.text_input("Enter your query here")

# LLM + Output parser
llm = ChatOpenAI(openai_api_key=openai_api_key, model="gpt-3.5-turbo")
parser = StrOutputParser()
chain = prompt | llm | parser

# Run the chain
if query:
    with st.spinner("Thinking..."):
        response = chain.invoke({"query": query})
        st.write(response)
