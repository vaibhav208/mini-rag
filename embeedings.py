# embeddings.py
import os
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Create embedding for a given text
def embed_text(text: str):
    response = client.embeddings.create(
        input=text,
        model="text-embedding-ada-002"
    )
    return response.data[0].embedding

# Split large text into overlapping chunks
def chunk_text(text, chunk_size=800, overlap=100):
    tokens = text.split()
    for i in range(0, len(tokens), chunk_size - overlap):
        yield " ".join(tokens[i:i + chunk_size])
