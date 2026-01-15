# PDF Chat Assistant - LangChain RAG Implementation

## Overview

PDF Chat Assistant is a production-ready **Retrieval-Augmented Generation (RAG)** system built with LangChain and Google Gemini AI. This project demonstrates advanced RAG architecture for document-based question answering with conversation memory and semantic search capabilities.

The system implements a complete RAG pipeline:

1. **Document Processing & Vectorization**
   * PDF text extraction using PyMuPDF for high-quality parsing
   * Recursive character-based text splitting (500 chars, 50 overlap)
   * Google Generative AI embeddings (gemini-embedding-001)
   * FAISS vector store for efficient similarity search
   * Automatic index creation and semantic retrieval

2. **RAG Chain with Memory**
   * LangChain VectorStoreRetriever for document retrieval
   * Context-aware prompting with chat history integration
   * Google Gemini Pro (gemini-pro-latest) for generation
   * Multi-turn conversation support with last 4 messages context
   * Source attribution distinguishing document vs. general knowledge

## RAG Architecture
```
┌─────────────┐
│  PDF Upload │
└──────┬──────┘
       │
       ▼
┌─────────────────┐
│ PyMuPDF Loader  │  ← langchain_community.document_loaders
└──────┬──────────┘
       │
       ▼
┌──────────────────────────┐
│ RecursiveCharacterText   │  ← langchain_text_splitters
│ Splitter (chunk_size=500)│
└──────┬───────────────────┘
       │
       ▼
┌─────────────────────────┐
│ GoogleGenerativeAI      │  ← langchain_google_genai
│ Embeddings              │
└──────┬──────────────────┘
       │
       ▼
┌─────────────────────────┐
│ FAISS VectorStore       │  ← langchain_community.vectorstores
│ (Semantic Index)        │
└──────┬──────────────────┘
       │
       ▼
┌─────────────────────────┐
│ VectorStoreRetriever    │  ← retriever.invoke(query)
│ (k=3, similarity search)│
└──────┬──────────────────┘
       │
       ▼
┌─────────────────────────┐
│ Context + Chat History  │  ← Prompt Engineering
│ + User Query            │
└──────┬──────────────────┘
       │
       ▼
┌─────────────────────────┐
│ ChatGoogleGenerativeAI  │  ← langchain_google_genai
│ (Gemini Pro)            │
└──────┬──────────────────┘
       │
       ▼
┌─────────────────────────┐
│ Generated Response      │
└─────────────────────────┘
```

## LangChain Components

### 1. Document Loaders
```python
from langchain_community.document_loaders import PyMuPDFLoader

loader = PyMuPDFLoader("document.pdf")
docs = loader.load()  # Returns List[Document]
```

### 2. Text Splitters
```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,        # Characters per chunk
    chunk_overlap=50       # Overlap for context continuity
)
chunks = splitter.split_documents(docs)
```

### 3. Embeddings
```python
from langchain_google_genai import GoogleGenerativeAIEmbeddings

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=GEMINI_API_KEY
)
```

### 4. Vector Store
```python
from langchain_community.vectorstores import FAISS

vectorstore = FAISS.from_documents(chunks, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
```

### 5. Retrieval
```python
# New LangChain Runnable interface
docs = retriever.invoke(query)  # Returns top-k relevant chunks
```

### 6. LLM Integration
```python
from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(
    model="models/gemini-pro-latest",
    google_api_key=GEMINI_API_KEY,
    temperature=0.2
)
response = llm.invoke(prompt)
```

## Features

* **Full LangChain RAG Pipeline**: Document loaders → Splitters → Embeddings → Vector Store → Retrieval → LLM
* **Conversation Memory**: Chat history integration for context-aware responses
* **Semantic Search**: FAISS vector similarity with top-3 retrieval
* **Chunking Strategy**: Recursive character splitting with overlap for context preservation
* **Source Attribution**: Distinguishes document-based vs. general knowledge answers
* **Efficient Retrieval**: FAISS indexing for fast similarity search
* **Streamlit Interface**: Interactive chat UI with session state management

## Technologies

**LangChain** | **Google Gemini AI** | **FAISS** | **PyMuPDF** | **Streamlit** | **RAG Architecture** | **Vector Embeddings** | **Semantic Search**

## API Key

Get your free Google Gemini API key:
[Google AI Studio](https://makersuite.google.com/app/apikey)
