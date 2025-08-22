# Generative AI Resume Assistant
This project is an AI-powered resume assistant and retrieval-augmented generation (RAG) system built with Python, 
Streamlit, LangChain, and ChromaDB. It allows users to upload their resume, ask questions about it, 
and get AI-generated answers using large language models (LLMs). and generate coverletters followign a predefiend template and guidance. (self defined)

## Features
- Upload your resume (PDF) and chat with it using LLMs (Ollama, Groq, LLaMA3, etc.)
- Q&A over resume content with semantic search and retrieval
- Resume bullet improvement and cover letter generation
- Uses vector database (Chroma) for fast document retrieval
- Streamlit web interface for easy interaction

## Technologies Used
- Python, Streamlit, LangChain, ChromaDB, HuggingFace Transformers, PyPDF, dotenv
- LLMs: Ollama, Groq, LLaMA3, LLaMA2
4. Upload your resume and start chatting!

## What Has Been Done & Detailed Process

### 1. Streamlit Resume Q&A App
- Built multiple Streamlit apps that let users upload a PDF resume and interact with it using LLMs.tested different models. 
- Used `PyPDFLoader` to extract text from uploaded PDFs.
- Split the extracted text into manageable chunks using `RecursiveCharacterTextSplitter` for better retrieval and embedding.
- Generated vector embeddings for each chunk using HuggingFace models
- Stored these embeddings in a Chroma vector database for fast semantic search.
- Implemented a retrieval-augmented generation (RAG) pipeline: when a user asks a question, the app retrieves the most relevant chunks from the database and passes them as context to
  the LLM (Ollama, Groq, LLaMA3, etc.) to generate an answer.
- Maintained chat history for a conversational experience.

### 2. Resume Bullet Improvement & Cover Letter Generation
- Added features to analyze and improve resume bullet points using LLMs (Groq/LLaMA3).
- Enabled automatic cover letter generation tailored to job descriptions and companies, leveraging the extracted resume content and user input.

### 3. Experimentation & Prototyping
- Used Jupyter notebooks to prototype and test document loading, chunking, embedding, and retrieval workflows.
- Explored different embedding models and chunking strategies for optimal retrieval performance.

### 4. API & Modularization
- Structured code into API and utility scripts for easier maintenance and extension.
- Used `.env` for secure API key management and ensured it is git-ignored.


- Created a `.gitignore` to keep sensitive files (like `.env`) out of version control.

---
Each step above is modular, allowing you to swap LLMs, embedding models, or vector DBs as needed. The project is designed for easy extension and experimentation with new retrieval or generation techniques.

Feel free to contribute or open issues!

