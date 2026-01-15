from langchain_community.document_loaders import PyMuPDFLoader

def load_pdf(file):
    with open("temp.pdf", "wb") as f:
        f.write(file.getbuffer())
    
    # PyMuPDF often has better text extraction
    loader = PyMuPDFLoader("temp.pdf")
    docs = loader.load()
    return docs