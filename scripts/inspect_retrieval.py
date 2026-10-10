
import os
import hashlib

from dotenv import load_dotenv
from langchain_pinecone import PineconeVectorStore

from src.helper import download_hugging_face_embeddings

load_dotenv()

INDEX_NAME = "medical-chatbot"
K = 5


def get_chunk_id(content: str) -> str:
    """Create a stable fingerprint for a text chunk."""
    normalized_content = " ".join(content.split())

    return hashlib.sha256(
        normalized_content.encode("utf-8")
    ).hexdigest()[:16]


def main():
    if not os.getenv("PINECONE_API_KEY"):
        raise ValueError(
            "PINECONE_API_KEY is missing from your .env file."
        )

    embeddings = download_hugging_face_embeddings()

    vector_store = PineconeVectorStore.from_existing_index(
        index_name=INDEX_NAME,
        embedding=embeddings,
    )

    queries = [
        "What are the symptoms of diabetes?",
        "What causes high blood pressure?",
        "What are common symptoms of anemia?",
    ]

    for query in queries:
        print("\n" + "=" * 75)
        print(f"QUESTION: {query}")
        print(f"K: {K}")
        print("=" * 75)

        results = vector_store.similarity_search_with_score(
            query,
            k=K,
        )

        if not results:
            print("No documents retrieved.")
            continue

        for rank, (document, score) in enumerate(results, start=1):
            print(f"\nRank: {rank}")
            print(f"Chunk ID: {get_chunk_id(document.page_content)}")
            print(f"Similarity score: {score}")
            print(f"Metadata: {document.metadata}")
            print(f"Content preview: {document.page_content[:500]!r}")


if __name__ == "__main__":
    main()
