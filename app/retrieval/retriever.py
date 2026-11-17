
import os

from dotenv import load_dotenv
from pinecone import Pinecone, DenseVectorQuery, TextQuery

from app.ingestion.embeddings import get_embedding_model

load_dotenv()

INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
API_KEY = os.getenv("PINECONE_API_KEY")

NAMESPACE = "employee-policy-v2"

DEFAULT_TOP_K = 3
CANDIDATE_TOP_K = 10
RRF_K = 60

INCLUDE_FIELDS = [
    "chunk_text",
    "title",
    "source",
    "file_type",
    "start_index",
]


def get_index():
    if not API_KEY or not INDEX_NAME:
        raise ValueError("Pinecone configuration missing from .env")

    pc = Pinecone(api_key=API_KEY)
    return pc.index(INDEX_NAME)


def validate_query(query: str):
    if not query or not query.strip():
        raise ValueError("Query cannot be empty")


def format_matches(matches, search_type: str):
    results = []

    for match in matches:
        results.append({
            "id": match.id,
            "content": match.get("chunk_text"),
            "title": match.get("title"),
            "source": match.get("source"),
            "file_type": match.get("file_type"),
            "start_index": match.get("start_index"),
            "dense_score": (
                match.score if search_type == "dense" else None
            ),
            "bm25_score": (
                match.score if search_type == "bm25" else None
            ),
            "rrf_score": None,
        })

    return results


def dense_search(query: str, top_k: int = DEFAULT_TOP_K):
    """Retrieve documents using semantic vector similarity."""

    validate_query(query)

    embedding_model = get_embedding_model()
    query_vector = embedding_model.embed_query(query)

    index = get_index()

    response = index.documents.search(
        namespace=NAMESPACE,
        top_k=top_k,
        score_by=[
            DenseVectorQuery(
                field="embedding",
                values=query_vector,
            )
        ],
        include_fields=INCLUDE_FIELDS,
    )

    return format_matches(response.matches, "dense")


def bm25_search(query: str, top_k: int = DEFAULT_TOP_K):
    """Retrieve documents using BM25 full-text relevance."""

    validate_query(query)

    index = get_index()

    response = index.documents.search(
        namespace=NAMESPACE,
        top_k=top_k,
        score_by=[
            TextQuery(
                query=query,
                fields=["chunk_text", "title"],
            )
        ],
        include_fields=INCLUDE_FIELDS,
    )

    return format_matches(response.matches, "bm25")


def hybrid_search(query: str, top_k: int = DEFAULT_TOP_K):
    """Combine dense and BM25 retrieval using Reciprocal Rank Fusion."""

    validate_query(query)

    dense_results = dense_search(
        query=query,
        top_k=CANDIDATE_TOP_K,
    )

    bm25_results = bm25_search(
        query=query,
        top_k=CANDIDATE_TOP_K,
    )

    fused_results = {}

    rankings = [
        ("dense", dense_results),
        ("bm25", bm25_results),
    ]

    for search_type, results in rankings:
        for rank, result in enumerate(results, start=1):

            document_id = result["id"]

            if document_id not in fused_results:
                fused_results[document_id] = {
                    **result,
                    "dense_score": None,
                    "bm25_score": None,
                    "rrf_score": 0.0,
                }

            fused_results[document_id]["rrf_score"] += (
                1 / (RRF_K + rank)
            )

            fused_results[document_id][
                f"{search_type}_score"
            ] = result[f"{search_type}_score"]

    results = sorted(
        fused_results.values(),
        key=lambda item: item["rrf_score"],
        reverse=True,
    )

    return results[:top_k]


def retrieve_documents(query: str, top_k: int = DEFAULT_TOP_K):
    """Backward-compatible default: hybrid retrieval."""

    return hybrid_search(query=query, top_k=top_k)
