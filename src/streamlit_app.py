import streamlit as st
import requests

st.set_page_config(page_title="AI Resume Analyzer", layout="centered")

st.title("📄 AI Resume Analyzer")
st.write("Upload your resume and get ATS score & suggestions")

uploaded_file = st.file_uploader(
    "Upload Resume (PDF only)",
    type=["pdf"]
)

if uploaded_file is not None:
    if st.button("Analyze Resume"):
        with st.spinner("Analyzing your resume..."):
            files = {
                "file": (uploaded_file.name, uploaded_file, "application/pdf")
            }

            response = requests.post(
                "https://ai-resume-backend.onrender.com/upload-resume",
                files=files
            )

        if response.status_code == 200:
            result = response.json()
            st.success("Analysis Complete ✅")

            st.subheader("📊 Result")
            st.json(result)
        else:
            st.error("❌ Analysis failed")
