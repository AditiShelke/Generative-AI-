import os
import streamlit as st
from dotenv import load_dotenv

# LangChain core
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.memory import ChatMessageHistory
from langchain.schema import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Retrieval extras
from langchain.retrievers import MultiQueryRetriever
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain.retrievers.document_compressors import CrossEncoderReranker

# LLMs
from langchain_groq import ChatGroq  # pip install langchain-groq
from langchain_community.chat_models import ChatOllama

# LangSmith (optional but supported)
from langsmith import Client
from langsmith.run_helpers import trace, traceable

load_dotenv()

# ---------- Config ----------
st.set_page_config(page_title="Resume Q&A (RAG)", page_icon="🧠", layout="wide")

PERSIST_DIR = "./chroma_resume_db"
EMB_MODEL = "intfloat/e5-base-v2"  # strong general-purpose retriever
CROSS_ENCODER = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # fast reranker
LS_PROJECT = os.getenv("LANGSMITH_PROJECT", "resume-rag-dev")
ls_client = Client()

# ---------- Sidebar ----------
with st.sidebar:
    st.header("Settings")
    provider = st.selectbox("LLM Provider", ["Groq (LLaMA3-70B)", "Ollama (local LLaMA3)"])
    temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.1)
    k = st.slider("Top-k after rerank", 2, 10, 4, 1)
    st.caption("Upload a resume PDF to (re)build the index.")
    st.markdown("---")
    st.caption("LangSmith tracing respects your LANGSMITH_* env vars.")

# ---------- Session State ----------
if "history" not in st.session_state:
    st.session_state.history = ChatMessageHistory()
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "resume_hash" not in st.session_state:
    st.session_state.resume_hash = "na"
if "last_trace_id" not in st.session_state:
    st.session_state.last_trace_id = None

# ---------- Helpers ----------
def get_llm():
    if provider.startswith("Groq"):
        api_key = os.getenv("GROQ_API_KEY", "")
        if not api_key:
            st.warning("GROQ_API_KEY not set; falling back to local Ollama.")
            return ChatOllama(model="llama3", temperature=temperature)
        return ChatGroq(model="llama3-70b-8192", temperature=temperature)
    else:
        # Requires: `ollama serve` and `ollama run llama3` in your terminal
        return ChatOllama(model="llama3", temperature=temperature)

def ensure_vectorstore(pdf_file=None):
    """Build or load Chroma index."""
    embed = HuggingFaceEmbeddings(
        model_name=EMB_MODEL,
        encode_kwargs={"normalize_embeddings": True}
    )
    if pdf_file is not None:
        # Save upload
        with open("temp_resume.pdf", "wb") as f:
            f.write(pdf_file.read())
        # Load + split
        docs = PyPDFLoader("temp_resume.pdf").load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)
        splits = splitter.split_documents(docs)
        # Store chunk_id in metadata for citation
        for i, d in enumerate(splits):
            md = d.metadata or {}
            md["chunk_id"] = i
            md["source"] = md.get("source", "resume.pdf")
            d.metadata = md
        # Persist
        vs = Chroma.from_documents(
            splits, embed, persist_directory=PERSIST_DIR, collection_name="resume"
        )
        vs.persist()
        st.session_state.resume_hash = str(abs(hash("".join([d.page_content[:50] for d in splits]))))
        st.success(f"Indexed {len(splits)} chunks into Chroma.")
        return vs
    # load existing
    if os.path.isdir(PERSIST_DIR):
        return Chroma(
            persist_directory=PERSIST_DIR,
            embedding_function=embed,
            collection_name="resume"
        )
    return None

def build_retriever(vs, llm):
    base = vs.as_retriever(search_kwargs={"k": max(6, k)})  # broaden before rerank
    # Multi-query improves recall
    mqr = MultiQueryRetriever.from_llm(retriever=base, llm=llm)
    # Cross-encoder rerank + compress to top-k
    reranker = HuggingFaceCrossEncoder(model_name=CROSS_ENCODER)
    compressor = CrossEncoderReranker(model=reranker, top_n=k)
    return ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=mqr,   # <-- important: base_retriever (not 'retriever')
    )

SYSTEM_PROMPT = """You are a helpful assistant that answers questions **only** using the provided resume context.
- If the answer is not in the context, say you don’t have enough information.
- Be concise and specific. Include inline citations like [#id] where id is the chunk index.
- Use short bullet points when helpful.
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    MessagesPlaceholder("history"),
    ("human", "Question: {question}\n\nContext:\n{context}\n\nAnswer with citations [#id].")
])

# LangSmith helpers
def lc_config(provider: str, model: str, resume_hash: str):
    return {
        "tags": ["resume-rag", provider, model],
        "metadata": {"provider": provider, "model": model, "resume_hash": resume_hash, "app": "streamlit"},
    }

@traceable(run_type="retriever", name="retrieve_resume_context")
def log_retrieval_for_trace(docs):
    # Return a serializable representation for LangSmith
    return [{"page_content": d.page_content, "metadata": d.metadata or {}} for d in docs]

def answer_question(llm, retriever, question):
    docs = retriever.get_relevant_documents(question)
    log_retrieval_for_trace(docs)
    blocks = []
    for i, d in enumerate(docs):
        cid = d.metadata.get("chunk_id", i)
        src = d.metadata.get("source", "resume.pdf")
        blocks.append(f"[#{cid}] ({src})\n{d.page_content.strip()}")
    context = "\n\n---\n\n".join(blocks) if blocks else "NO CONTEXT"
    chain = (prompt | llm)
    resp = chain.invoke({
        "history": st.session_state.history.messages,
        "question": question,
        "context": context
    })
    # resp is AIMessage for chat models
    return (resp.content if hasattr(resp, "content") else str(resp)), docs

# ---------- UI ----------
st.title("🧠 Resume Q&A — RAG over your PDF")

uploaded = st.file_uploader("Upload resume PDF", type=["pdf"])
c1, c2 = st.columns(2)
with c1:
    if st.button("Build / Rebuild Index", use_container_width=True, type="primary"):
        if uploaded is None:
            st.warning("Please upload a PDF first.")
        else:
            st.session_state.vectorstore = ensure_vectorstore(uploaded)
with c2:
    if st.button("Load Existing Index", use_container_width=True):
        st.session_state.vectorstore = ensure_vectorstore(None)
        if st.session_state.vectorstore:
            st.success("Loaded existing Chroma index.")
        else:
            st.info("No existing index found; upload a PDF to build one.")

llm = get_llm()
retriever = build_retriever(st.session_state.vectorstore, llm) if st.session_state.vectorstore else None

st.markdown("### Ask a question about the resume")
q = st.text_input("e.g., “What projects show experience with RAG and Streamlit?”")

if st.button("Ask", disabled=(retriever is None or not q.strip()), type="secondary"):
    if retriever is None:
        st.error("Index not loaded.")
    else:
        st.session_state.history.add_user_message(q)

        cfg = lc_config(
            provider="groq" if provider.startswith("Groq") else "ollama",
            model="llama3-70b-8192" if provider.startswith("Groq") else "llama3",
            resume_hash=st.session_state.resume_hash,
        )

        with trace(name="ask_resume_question", inputs={"question": q}, project_name=LS_PROJECT) as root:
            # run
            answer, docs = answer_question(
                llm.with_config(cfg), retriever, q
            )
            root.outputs = {"answer": answer}
            st.session_state.last_trace_id = root.id

        st.session_state.history.add_ai_message(answer)
        st.subheader("Answer")
        st.write(answer)

        with st.expander("Show retrieved context"):
            for d in docs:
                cid = d.metadata.get("chunk_id", "?")
                st.markdown(f"**Chunk [#{cid}]** — {d.metadata.get('source','resume.pdf')}")
                st.code(d.page_content.strip())

        # Feedback buttons to LangSmith
        fb1, fb2 = st.columns(2)
        with fb1:
            if st.button("👍 Helpful"):
                if st.session_state.last_trace_id:
                    ls_client.create_feedback(key="user_rating", score=1, trace_id=st.session_state.last_trace_id)
                st.toast("Thanks! Logged 👍.")
        with fb2:
            if st.button("👎 Not helpful"):
                if st.session_state.last_trace_id:
                    ls_client.create_feedback(key="user_rating", score=0, trace_id=st.session_state.last_trace_id)
                st.toast("Logged 👎.")

st.markdown("### Chat History")
for m in st.session_state.history.messages[-8:]:
    role = "You" if isinstance(m, HumanMessage) else "Assistant"
    st.markdown(f"**{role}:** {m.content}")
