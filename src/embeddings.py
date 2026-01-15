from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import os

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

def create_vectorstore(chunks):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=GEMINI_API_KEY
    )
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore
