
import os
import json
import hashlib
from pathlib import Path
from statistics import mean

from dotenv import load_dotenv
from langchain_pinecone import PineconeVectorStore

from src.helper import download_hugging_face_embeddings
from src.retrieval_metrics import (
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)

load_dotenv()

INDEX_NAME = "medical-chatbot"
K_VALUES = [3, 5, 10]

DATASET_PATH = Path("data/evaluation/retrieval_dataset.json")


def get_chunk_id(content: str) -> str:
    """Keep this identical to the inspection script."""
    normalized_content = " ".join(content.split())

    return hashlib.sha256(
        normalized_content.encode("utf-8")
    ).hexdigest()[:16]


def load_dataset():
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_PATH}"
        )

    with DATASET_PATH.open("r", encoding="utf-8") as file:
        dataset = json.load(file)

    if not dataset:
        raise ValueError("The evaluation dataset is empty.")

    for index, item in enumerate(dataset, start=1):
        question = item.get("question", "").strip()
        relevant_ids = item.get("relevant_document_ids")

        if not question:
            raise ValueError(
                f"Dataset entry {index} has no question."
            )

        if not isinstance(relevant_ids, list) or not relevant_ids:
            raise ValueError(
                f"Entry {index} needs verified relevant_document_ids."
            )

        if not all(
            isinstance(chunk_id, str) and chunk_id.strip()
            for chunk_id in relevant_ids
        ):
            raise ValueError(
                f"Entry {index} contains an invalid chunk ID."
            )

    return dataset


def retrieve_documents(vector_store, question, k):
    """Return retrieved documents and elapsed latency."""
    from time import perf_counter

    start = perf_counter()
    documents = vector_store.similarity_search(
        question,
        k=k,
    )
    elapsed_ms = (perf_counter() - start) * 1000

    return documents, elapsed_ms


def main():
    if not os.getenv("PINECONE_API_KEY"):
        raise ValueError(
            "PINECONE_API_KEY is missing from your .env file."
        )

    dataset = load_dataset()

    embeddings = download_hugging_face_embeddings()

    vector_store = PineconeVectorStore.from_existing_index(
        index_name=INDEX_NAME,
        embedding=embeddings,
    )

    results_by_k = {k: [] for k in K_VALUES}

    for item in dataset:
        question = item["question"]
        relevant_ids = set(item["relevant_document_ids"])

        documents, latency_ms = retrieve_documents(
            vector_store,
            question,
            max(K_VALUES),
        )

        retrieved_ids = [
            get_chunk_id(document.page_content)
            for document in documents
        ]

        print("\n" + "-" * 75)
        print(f"Question: {question}")
        print(f"Retrieved: {len(retrieved_ids)} chunks")
        print(f"Top-10 retrieval latency: {latency_ms:.2f} ms")

        for k in K_VALUES:
            top_k = retrieved_ids[:k]

            precision = precision_at_k(
                top_k, relevant_ids, k
            )
            recall = recall_at_k(
                top_k, relevant_ids, k
            )
            rr = reciprocal_rank(
                top_k, relevant_ids
            )

            results_by_k[k].append(
                {
                    "precision": precision,
                    "recall": recall,
                    "rr": rr,
                    "latency_ms": latency_ms,
                }
            )

    print("\n" + "=" * 85)
    print("RETRIEVAL BENCHMARK RESULTS")
    print("=" * 85)
    print(
        f"{'K':<6}"
        f"{'Precision@K':<16}"
        f"{'Recall@K':<14}"
        f"{'MRR':<12}"
        f"{'Mean latency (ms)':<20}"
    )

    for k in K_VALUES:
        rows = results_by_k[k]

        print(
            f"{k:<6}"
            f"{mean(r['precision'] for r in rows):<16.4f}"
            f"{mean(r['recall'] for r in rows):<14.4f}"
            f"{mean(r['rr'] for r in rows):<12.4f}"
            f"{mean(r['latency_ms'] for r in rows):<20.2f}"
        )


if __name__ == "__main__":
    main()
