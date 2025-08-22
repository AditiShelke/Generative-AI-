import requests
import streamlit as st
import PyPDF2

# --- Extract text from uploaded PDF ---
def extract_pdf_text(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() or ""
    return text.strip()

# --- Call Groq API for resume bullet improvement ---
def get_groq_resume_response(resume_text):
    try:
        response = requests.post(
            "http://localhost:8000/groq/resume/invoke",
            json={"input": {"input": resume_text}}
        )
        response.raise_for_status()
        return response.json().get("output", "⚠️ No output returned.")
    except Exception as e:
        return f"❌ Error from Groq API: {e}"

# --- Call Ollama (LLaMA2) API for cover letter generation ---
def get_llama2_coverletter_response(resume_text, job_desc, company, style_guide):
    try:
        response = requests.post(
            "http://localhost:8000/llama2/coverletter/invoke",
            json={"input": {
                "resume": resume_text,
                "job": job_desc,
                "company": company,
                "style_guide": style_guide
            }}
        )
        response.raise_for_status()
        return response.json().get("output", "⚠️ No output returned.")
    except Exception as e:
        return f"❌ Error from LLaMA2 API: {e}"

# --- Streamlit UI ---
st.title("📄 Smart Resume Assistant with Style Templates")
st.markdown("Upload your resume, input job details, and optionally provide a cover letter style guide.")

# Inputs
pdf_file = st.file_uploader("📎 Upload your resume (PDF only)", type="pdf")
job_desc = st.text_area("💼 Paste the job description")
company = st.text_input("🏢 Company name")
style_file = st.file_uploader("🎨 (Optional) Upload cover letter style guide (TXT or PDF)", type=["txt", "pdf"])

# Parse style guide
style_guide = ""
if style_file:
    if style_file.type == "application/pdf":
        style_guide = extract_pdf_text(style_file)
    else:
        style_guide = style_file.read().decode("utf-8")

# Resume processing
if pdf_file:
    resume_text = extract_pdf_text(pdf_file)
    st.subheader("📃 Extracted Resume Content")
    st.text_area("Resume Preview", resume_text, height=200)

    # Resume bullet improvement via Groq
    if st.button("✏️ Improve Bullet Points with Groq (LLaMA3)"):
        with st.spinner("Analyzing with Groq..."):
            result = get_groq_resume_response(resume_text)
            st.subheader("✅ Suggested Improvements")
            st.write(result)

    # Cover letter generation via LLaMA2
    if job_desc and company and st.button("📬 Generate Cover Letter with LLaMA2"):
        with st.spinner("Crafting your cover letter..."):
            result = get_llama2_coverletter_response(resume_text, job_desc, company, style_guide)
            st.subheader("📄 Your Tailored Cover Letter")
            st.write(result)
