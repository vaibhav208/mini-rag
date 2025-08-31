from PyPDF2 import PdfReader
import os
from sentence_transformers import SentenceTransformer
from db import index

# Load HuggingFace embedding model (local, no API needed)
model = SentenceTransformer("all-MiniLM-L6-v2")

def load_pdf(path):
    """Extract text from PDF file"""
    reader = PdfReader(path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

def embed_texts(texts):
    """Convert list of texts into embeddings"""
    return model.encode(texts)

def add_documents(texts):
    """Add documents to FAISS"""
    vectors = embed_texts(texts)
    index.add(vectors, texts)

def ingest_file(path: str):
    """Ingest a document (PDF or TXT) into FAISS"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"❌ File not found: {path}")

    if path.endswith(".pdf"):
        raw_text = load_pdf(path)
    elif path.endswith(".txt"):
        with open(path, "r", encoding="utf-8") as f:
            raw_text = f.read()
    else:
        raise ValueError("❌ Only PDF and TXT supported")

    # Split into chunks (optional: here we just keep as one block)
    chunks = [raw_text[i:i+500] for i in range(0, len(raw_text), 500)]

    # Add to FAISS
    add_documents(chunks)
    print(f"✅ Ingested {len(chunks)} chunks from {path}")
