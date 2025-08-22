# build_style_vectorstore.py
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# Load your writing samples
loader = TextLoader("data/aditi_writing_samples.txt")
docs = loader.load()

# Split & embed
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(docs)
db = FAISS.from_documents(chunks, HuggingFaceEmbeddings())
db.save_local("embeddings/aditi_style_vectorstore")
