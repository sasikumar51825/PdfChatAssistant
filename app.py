import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from src.loaders import load_pdf
from src.splitters import split_docs
from src.embeddings import create_vectorstore
from src.llm import get_gemini_llm
from src.chains import build_rag_chain

st.title("ChatPDF - Talk to Your Documents")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None

uploaded_file = st.file_uploader("Upload PDF", type="pdf")

if uploaded_file and not st.session_state.rag_chain:
    with st.spinner("Processing..."):
        docs = load_pdf(uploaded_file)
        chunks = split_docs(docs)
        vectorstore = create_vectorstore(chunks)
        llm = get_gemini_llm()
        st.session_state.rag_chain = build_rag_chain(llm, vectorstore)

if st.session_state.rag_chain:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
    
    if prompt := st.chat_input("Ask about the document"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)
        
        with st.chat_message("assistant"):
            # Pass chat history to rag_chain
            response = st.session_state.rag_chain(prompt, st.session_state.messages[:-1])
            st.write(response)
        st.session_state.messages.append({"role": "assistant", "content": response})