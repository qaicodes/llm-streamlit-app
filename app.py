import streamlit as st
from PyPDF2 import PdfReader
from llm import generative_llm
def process_pdf_and_answer(file, question):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return your_backend_function(text, question)  # Call your function directly

generative_llm("path", "question")

st.title("LEGAL GenAI")

uploaded_file = st.file_uploader("Choose a PDF or text file", type=["pdf", "txt"])
question = st.text_input("Ask your question about the document:")

if uploaded_file and question:
    if uploaded_file.type == "application/pdf":
        answer = process_pdf_and_answer(uploaded_file, question)
    else:  # It's a text file
        text = uploaded_file.read().decode()
        answer = your_backend_function(text, question)

    st.write("Answer:")
    st.write(answer)  # Directly display the answer 
