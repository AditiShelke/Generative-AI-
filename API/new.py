import fitz  # PyMuPDF
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import StrOutputParser
from langchain.chat_models import ChatOpenAI
from langchain_community.llms import Ollama

def extract_pdf_text(pdf_file):
    with fitz.open(stream=pdf_file.read(), filetype="pdf") as doc:
        text = ""
        for page in doc:
            text += page.get_text()
    return text


prompt = ChatPromptTemplate.from_messages([
    ("system", "You're a resume assistant. Analyze these bullet points and suggest improvements:\n\n{resume_text}")
])
chain = prompt | llm | StrOutputParser()

prompt = ChatPromptTemplate.from_messages([
    ("system", "You're a resume assistant. Analyze these bullet points and suggest improvements:\n\n{resume_text}")
])
chain = prompt | llm | StrOutputParser()
