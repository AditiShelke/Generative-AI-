import os
import streamlit as st
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.memory import ChatMessageHistory
from langchain.schema import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.retrievers import MultiQueryRetriever
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain.retrievers.document_compressors import CrossEncoderReranker

# LLMs
from langchain_groq import ChatGroq    # pip install langchain-groq
from langchain_community.chat_models import ChatOllama

load_dotenv()

st.set_page_config(page_title="Resume Q&A (RAG)", page_icon="🧠", layout="wide")

# ---------- Config ----------
PERSIST_DIR = "./chroma_resume_db"
EMB_MODEL = "intfloat/e5-base-v2"  # strong general retriever; normalized embeddings
CROSS_ENCODER = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # fast reranker

# ---------- Sidebar ----------
with st.sidebar:
    st.header("Settings")
    provider = st.selectbox("LLM Provider", ["Groq (LLaMA3)", "Ollama (local LLaMA3)"])
    temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.1)
    k = st.slider("Top-k retrieve", 2, 10, 4, 1)
    st.markdown("---")
    st.caption("Upload a resume PDF to (re)build the index.")

# ---------- Session State ----------
if "history" not in st.session_state:
    st.session_state.history = ChatMessageHistory()

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

# ---------- Helpers ----------
def get_llm():
    if provider.startswith("Groq"):
        api_key = os.getenv("GROQ_API_KEY", "")
        if not api_key:
            st.warning("GROQ_API_KEY not set; falling back to Ollama.")
            return ChatOllama(model="llama3", temperature=temperature)
        return ChatGroq(model="llama3-70b-8192", temperature=temperature)
    else:
        # Requires: `ollama serve` and `ollama run llama3`
        return ChatOllama(model="llama3", temperature=temperature)

def ensure_vectorstore(pdf_file=None):
    """Build or load Chroma index."""
    embed = HuggingFaceEmbeddings(
        model_name=EMB_MODEL,
        encode_kwargs={"normalize_embeddings": True}
    )
    if pdf_file is not None:
        with open("temp_resume.pdf", "wb") as f:
            f.write(pdf_file.read())
        loader = PyPDFLoader("temp_resume.pdf")
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)
        splits = splitter.split_documents(docs)
        vs = Chroma.from_documents(
            splits, embed, persist_directory=PERSIST_DIR, collection_name="resume"
        )
        vs.persist()
        st.success("Indexed resume into Chroma.")
        return vs
    # try loading existing
    if os.path.isdir(PERSIST_DIR):
        return Chroma(persist_directory=PERSIST_DIR, embedding_function=embed, collection_name="resume")
    return None

def build_retriever(vs, llm):
    base = vs.as_retriever(search_kwargs={"k": max(4, k)})
    # Multi-query to expand recall
    mqr = MultiQueryRetriever.from_llm(retriever=base, llm=llm)
    # Rerank + compress to keep only best passages
    reranker = HuggingFaceCrossEncoder(model_name=CROSS_ENCODER)
    compressor = CrossEncoderReranker(model=reranker, top_n=k)
    compressed = ContextualCompressionRetriever(base_compressor=compressor, retriever=mqr)
    return compressed

SYSTEM_PROMPT = """You are a helpful assistant that answers questions **only** using the provided resume context.
- If the answer is not in the context, say you don’t have enough information.
- Be concise and specific. When listing skills/experience, cite the matching snippet IDs.
- Output markdown with short bullet points when helpful.
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    MessagesPlaceholder("history"),
    ("human", "Question: {question}\n\nContext:\n{context}\n\nAnswer with citations [#id] where id is the source chunk index.")
])

def answer_question(llm, retriever, question):
    # Retrieve docs
    docs = retriever.get_relevant_documents(question)
    context_blocks = []
    for i, d in enumerate(docs):
        # Attach short ids for inline citations
        meta = d.metadata or {}
        src = meta.get("source", "resume.pdf")
        context_blocks.append(f"[#{i}] ({src})\n{d.page_content.strip()}")
    context = "\n\n---\n\n".join(context_blocks) if context_blocks else "NO CONTEXT"
    # Build history
    history_msgs = []
    for m in st.session_state.history.messages:
        history_msgs.append(m)
    chain = prompt | llm
    resp = chain.invoke({"history": history_msgs, "question": question, "context": context})
    return resp.content, docs

# ---------- UI ----------
st.title("🧠 Resume Q&A — RAG over your PDF")

uploaded = st.file_uploader("Upload resume PDF", type=["pdf"])
col1, col2 = st.columns([1,1])
with col1:
    if st.button("Build / Rebuild Index", use_container_width=True, type="primary"):
        if uploaded is None:
            st.warning("Please upload a PDF first.")
        else:
            st.session_state.vectorstore = ensure_vectorstore(uploaded)
with col2:
    if st.button("Load Existing Index", use_container_width=True):
        st.session_state.vectorstore = ensure_vectorstore(None)
        if st.session_state.vectorstore:
            st.success("Loaded existing Chroma index.")
        else:
            st.info("No existing index found; upload a PDF to build one.")

llm = get_llm()
if st.session_state.vectorstore:
    retriever = build_retriever(st.session_state.vectorstore, llm)
else:
    retriever = None

st.markdown("### Ask a question about the resume")
q = st.text_input("e.g., “What projects show experience with RAG and Streamlit?”")

if st.button("Ask", disabled=retriever is None or not q.strip(), type="secondary"):
    if retriever is None:
        st.error("Index not loaded. Upload/build or load existing index.")
    else:
        st.session_state.history.add_message(HumanMessage(content=q))
        answer, docs = answer_question(llm, retriever, q)
        st.session_state.history.add_message(AIMessage(content=answer))
        st.markdown("#### Answer")
        st.write(answer)
        with st.expander("Show retrieved context"):
            for i, d in enumerate(docs):
                st.markdown(f"**Chunk #{i}** — {d.metadata.get('source','resume.pdf')}")
                st.code(d.page_content.strip())

st.markdown("### Chat History")
for m in st.session_state.history.messages[-8:]:
    role = "You" if isinstance(m, HumanMessage) else "Assistant"
    st.markdown(f"**{role}:** {m.content}")
