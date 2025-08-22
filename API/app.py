# app.py — FastAPI backend for resume assistant

from fastapi import FastAPI
from langchain_groq import ChatGroq
#chatopenai acts up (crying emoji) cant debug further no energy left
# this is only note to self because i happen to forget and take time before i come back : but i tried doing this thing where i combine Groq + LLaMA 3 (patience ran out)
#works fine for the resume points but takes time forever for coverletter 
########################################################### work on thiss later#############################################################################
from langchain_openai import ChatOpenAI
from langchain_community.llms import Ollama
from langchain.prompts import ChatPromptTemplate
from langserve import add_routes
import uvicorn
import os
from dotenv import load_dotenv
from fastapi import FastAPI
from langchain_openai import ChatOpenAI
from langchain_community.llms import Ollama
from langchain.prompts import ChatPromptTemplate
from langserve import add_routes
import uvicorn
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")
if openai_key:
    os.environ["OPENAI_API_KEY"] = openai_key
else:
    raise EnvironmentError("❌ OPENAI_API_KEY is missing from your .env file")

# Initialize FastAPI app
app = FastAPI(
    title="LLM Resume Assistant API",
    description="API to analyze resume content and generate personalized cover letters",
    version="1.0"
)

# Set up LLMs
# openai_llm = ChatOpenAI(model="gpt-3.5-turbo")
groq_api_key = os.getenv("GROQ_API_KEY")
groq_llm = ChatGroq(groq_api_key=groq_api_key, model="llama3-8b-8192")

ollama_llm = Ollama(model="llama2")  # Ensure llama2 is running with `ollama run llama2`

# Prompt for resume bullet point improvements
resume_prompt = ChatPromptTemplate.from_template(
    "You are a resume assistant. Follow these bullet point writing rules:\n"
    "- Start with strong action verbs\n"
    "- Be concise and quantified (e.g., increased X by Y%)\n"
    "- Avoid passive voice and generic statements\n"
    "- Match the tone to technical/data science roles\n\n"
    "Now analyze and improve this resume:\n\n{input}"
)


cover_prompt = ChatPromptTemplate.from_template(
    "You are a career coach. Write a cover letter using this template:\n"
    "1. Introduction: State the role and enthusiasm.\n"
    "2. Why me: 2-3 bullet-style achievements aligned with job.\n"
    "3. Why them: Mention company impact or values and how i can contribute given my experience and knowledge area aligns with their need or to help them scale up.\n"
    "4. Conclusion: Call to action.\n\n"
    "Resume:\n{resume}\n\nJob Description:\n{job}\n\nCompany:\n{company}"
)

# LangServe routes
# add_routes(app, resume_prompt | openai_llm, path="/openai/resume")
add_routes(app, resume_prompt | groq_llm, path="/groq/resume")
add_routes(app, cover_prompt | ollama_llm, path="/llama2/coverletter")


# Run the app
if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000, log_level="info")

