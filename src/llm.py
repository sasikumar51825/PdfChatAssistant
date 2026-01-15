from langchain_google_genai import ChatGoogleGenerativeAI
import os

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

def get_gemini_llm():
    llm = ChatGoogleGenerativeAI(
        model="models/gemini-pro-latest",
        google_api_key=GEMINI_API_KEY,
        temperature=0.2
    )
    return llm
