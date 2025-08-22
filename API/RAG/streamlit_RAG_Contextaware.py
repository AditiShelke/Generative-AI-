# streamlit_resume_chat.py

import streamlit as st
from dotenv import load_dotenv
import os

from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

# Load environment variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# Page config
st.set_page_config(page_title="📄 Multi-turn Resume Chat (Groq + LLaMA3)")
st.title("📄 Chat With My Resume — Multi-turn")

# Sidebar
st.sidebar.markdown("### Instructions")
st.sidebar.markdown("Upload your resume as `.pdf`, and start a multi-turn chat with it!")

# Upload resume
uploaded_file = st.file_uploader("Upload your resume (.pdf)", type=["pdf"])

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# When file is uploaded
if uploaded_file is not None:
    with open("temp_resume.pdf", "wb") as f:
        f.write(uploaded_file.read())

    # Load and chunk
    loader = PyPDFLoader("temp_resume.pdf")
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    # Embedding and DB
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = Chroma.from_documents(chunks, embedding_model, persist_directory="chroma_resume_db")
    retriever = db.as_retriever()

    # LLM
    llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama3-70b-8192")

    # Memory for multi-turn
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    # Conversational RAG chain
    qa_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever, memory=memory)

    # Chat input
    user_query = st.chat_input("Ask something about your resume (multi-turn supported)...")

    if user_query:
        with st.spinner("Answering..."):
            response = qa_chain.run(user_query)
            st.session_state.chat_history.append((user_query, response))

    # Display chat
    for q, a in st.session_state.chat_history:

        st.chat_message("user").write(q)
        st.chat_message("ai").write(a)
