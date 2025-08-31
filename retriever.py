import numpy as np
from db import index

def retrieve(query: str, top_k: int = 3):
    """Retrieve top-k chunks from FAISS index (local only)."""
    # Convert query into a fake embedding (very simple for now)
    vector = np.array([ord(c) for c in query[:512]]).astype("float32")

    # Search in FAISS index
    distances, indices = index.search(np.expand_dims(vector, axis=0), top_k)

    results = []
    for idx in indices[0]:
        if 0 <= idx < len(index.texts):
            results.append(index.texts[idx])
    return results
