import os
import faiss
import numpy as np
import pickle

# ✅ Disable Pinecone (local only)
use_pinecone = False

# Files for saving FAISS index and texts
faiss_index_file = "faiss_index.bin"
texts_file = "texts.pkl"

# Embedding dimension (must match your SentenceTransformer model, e.g., all-MiniLM-L6-v2 = 384)
embedding_dim = 384

# Load or create FAISS index
if os.path.exists(faiss_index_file) and os.path.exists(texts_file):
    print("🔄 Loading existing FAISS index...")
    index = faiss.read_index(faiss_index_file)
    with open(texts_file, "rb") as f:
        texts = pickle.load(f)
else:
    print("✨ Creating new FAISS index...")
    index = faiss.IndexFlatL2(embedding_dim)
    texts = []

# Save function (keeps index persistent between runs)
def save_index():
    faiss.write_index(index, faiss_index_file)
    with open(texts_file, "wb") as f:
        pickle.dump(texts, f)
    print("💾 FAISS index saved successfully.")
