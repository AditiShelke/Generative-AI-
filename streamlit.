import streamlit as st
from dotenv import load_dotenv
import os

from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA

# Load API key from .env
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# Streamlit app config
st.set_page_config(page_title="📄 Chat With My Resume (Groq + LLaMA3)")
st.title("📄 Chat With My Resume")

st.sidebar.header("Instructions")
st.sidebar.info("Upload your resume in **PDF** format and ask questions about it using LLaMA3 via Groq!")

# File upload
uploaded_file = st.file_uploader("Upload your resume (PDF only)", type=["pdf"])

if uploaded_file is not None:
    with st.spinner("Reading and indexing your resume..."):
        # Save the uploaded PDF temporarily
        with open("Aditi_ShelkeResume.pdf", "wb") as f:
            f.write(uploaded_file.read())

        # Load and process
        loader = PyPDFLoader("Aditi_ShelkeResume.pdf")
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_documents(docs)

        # Embeddings
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        db = Chroma.from_documents(chunks, embeddings, persist_directory="chroma_resume_db")
        retriever = db.as_retriever()

        # LLM from Groq
        llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama3-70b-8192")

        # RAG QA chain
        qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

    # Chat UI
    st.success("✅ Resume processed! Now ask me anything about it:")

    user_query = st.text_input("Ask a question about your resume:", placeholder="e.g. What are my key skills?")
    if user_query:
        with st.spinner("Running Running ..."):
            answer = qa.run(user_query)
            st.write("🤖", answer)
