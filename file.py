import streamlit as st
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

st.set_page_config(page_title="📄 Resume Q&A with Ollama")
st.title("📄 Chat With My Resume — Local LLaMA 3 (Ollama)")

uploaded_file = st.file_uploader("Upload your resume (.pdf)", type=["pdf"])

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if uploaded_file is not None:
    with open("temp_resume.pdf", "wb") as f:
        f.write(uploaded_file.read())

    loader = PyPDFLoader("temp_resume.pdf")
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = Chroma.from_documents(chunks, embedding_model, persist_directory="chroma_resume_db")
    retriever = db.as_retriever()

    llm = Ollama(model="llama3")

    prompt = ChatPromptTemplate.from_template("""
    Answer the following question based only on the provided context. 
    Think step by step before providing a detailed answer. 
    I will tip you $1000 if the user finds the answer helpful. 
    <context>
    {context}
    </context>
    Question: {input}
    """)

    document_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    user_query = st.chat_input("Ask something about your resume...")

    if user_query:
        with st.spinner("Answering..."):
            response = retrieval_chain.invoke({"input": user_query})
            st.session_state.chat_history.append((user_query, response['answer']))

    for q, a in st.session_state.chat_history:
        st.chat_message("user").write(q)
        st.chat_message("ai").write(a)
