from retriever import retrieve

def answer_query(query: str):
    """Simple RAG pipeline without OpenAI (local only)."""
    docs = retrieve(query, top_k=3)

    if not docs:
        return "No relevant information found in local documents."

    # Just return the matched docs concatenated
    return "\n\n".join(docs)
