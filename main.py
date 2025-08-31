from fastapi import FastAPI
from rag import answer_query

app = FastAPI()

@app.get("/")
def root():
    return {
        "message": "Me-API Playground is running",
        "try": ["/health", "/ask?query=hello", "/docs"]
    }

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/ask")
def ask(query: str):
    answer = answer_query(query)
    return {"query": query, "answer": answer}
