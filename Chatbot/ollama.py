from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import streamlit as st
import os
from dotenv import load_dotenv
#takes time to reply 
# can set queries limit and other atrributes to speed up the response 
# Load environment variables
load_dotenv()

# Optional: LangSmith Tracking
if os.getenv("LANGCHAIN_API_KEY"):
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
    os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGSMITH_PROJECT", "resume-assistant")
    os.environ["LANGCHAIN_ENDPOINT"] = os.getenv("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")

# Create prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Please respond to the user's queries."),
    ("user", "Question: {question}")
])

# Load LLaMA model from Ollama
llm = Ollama(model="llama3.2:latest")

# Chain setup
output_parser = StrOutputParser()
chain = prompt | llm | output_parser

# Streamlit UI
st.title("LangChain + Ollama Chatbot")

user_input = st.text_input("Ask me anything!")

if user_input:
    with st.spinner("Thinking..."):
        try:
            response = chain.invoke({"question": user_input})
            st.success("Answer:")
            st.write(response)
        except Exception as e:
            st.error(f"Error: {e}")
