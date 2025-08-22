from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import os, shutil
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings

app = FastAPI()
PERSIST_DIR = "./chroma_resume_db"
EMB_MODEL = "intfloat/e5-base-v2"

class AskRequest(BaseModel):
    question: str

@app.post("/index")
async def index_resume(file: UploadFile = File(...)):
    path = "temp_resume.pdf"
    with open(path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    docs = PyPDFLoader(path).load()
    splits = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120).split_documents(docs)
    embed = HuggingFaceEmbeddings(model_name=EMB_MODEL, encode_kwargs={"normalize_embeddings": True})
    vs = Chroma.from_documents(splits, embed, persist_directory=PERSIST_DIR, collection_name="resume")
    vs.persist()
    return {"status": "ok", "chunks": len(splits)}

@app.post("/ask")
async def ask(req: AskRequest):
    embed = HuggingFaceEmbeddings(model_name=EMB_MODEL, encode_kwargs={"normalize_embeddings": True})
    vs = Chroma(persist_directory=PERSIST_DIR, embedding_function=embed, collection_name="resume")
    docs = vs.similarity_search(req.question, k=6)
    # Return raw contexts; Streamlit can call Groq/Ollama and render nicely
    return {"contexts": [d.page_content for d in docs], "metadatas": [d.metadata for d in docs]}
