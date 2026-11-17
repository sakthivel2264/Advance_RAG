
import json
from pathlib import Path
from statistics import mean

from app.retrieval.retriever import (
    dense_search,
    bm25_search,
    hybrid_search,
)

from app.evaluation.retrieval_metrics import (
    precision_at_k,
    precision_at_retrieved,
    recall_at_k,
    hit_rate_at_k,
    reciprocal_rank,
    mean_reciprocal_rank,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "rag_test_dataset.json"
)

K = 3

RETRIEVAL_METHODS = {
    "Dense": dense_search,
    "BM25": bm25_search,
    "Hybrid": hybrid_search,
}


def load_dataset():
    with open(DATASET_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


def evaluate_method(method_name, retrieval_function, dataset):

    precision_scores = []
    actual_precision_scores = []
    recall_scores = []
    hit_rate_scores = []
    reciprocal_ranks = []

    print(f"\n========== {method_name.upper()} RETRIEVAL ==========\n")

    for test_case in dataset:

        question = test_case["question"]
        relevant_ids = test_case["relevant_document_ids"]

        retrieved_documents = retrieval_function(
            query=question,
            top_k=K,
        )

        retrieved_ids = [
            document["id"]
            for document in retrieved_documents
        ]

        precision = precision_at_k(
            retrieved_ids, relevant_ids, K
        )

        actual_precision = precision_at_retrieved(
            retrieved_ids, relevant_ids, K
        )

        recall = recall_at_k(
            retrieved_ids, relevant_ids, K
        )

        hit_rate = hit_rate_at_k(
            retrieved_ids, relevant_ids, K
        )

        rr = reciprocal_rank(
            retrieved_ids, relevant_ids
        )

        precision_scores.append(precision)
        actual_precision_scores.append(actual_precision)
        recall_scores.append(recall)
        hit_rate_scores.append(hit_rate)
        reciprocal_ranks.append(rr)

        print(f"Test case: {test_case['id']}")
        print(f"Question: {question}")
        print(f"Retrieved: {len(retrieved_ids)}")
        print(f"Precision@{K}: {precision:.4f}")
        print(f"Actual-result precision: {actual_precision:.4f}")
        print(f"Recall@{K}: {recall:.4f}")
        print(f"Hit Rate@{K}: {hit_rate:.4f}")
        print(f"Reciprocal Rank: {rr:.4f}")
        print("-" * 50)

    return {
        "precision": mean(precision_scores),
        "actual_precision": mean(actual_precision_scores),
        "recall": mean(recall_scores),
        "hit_rate": mean(hit_rate_scores),
        "mrr": mean_reciprocal_rank(reciprocal_ranks),
    }


def run_evaluation():

    dataset = load_dataset()

    if not dataset:
        raise ValueError("Evaluation dataset is empty")

    benchmark_results = {}

    for method_name, retrieval_function in RETRIEVAL_METHODS.items():

        benchmark_results[method_name] = evaluate_method(
            method_name,
            retrieval_function,
            dataset,
        )

    print("\n========== RETRIEVAL BENCHMARK ==========\n")

    print(f"{'Metric':<28} {'Dense':>10} {'BM25':>10} {'Hybrid':>10}")
    print("-" * 62)

    metrics = [
        ("Precision@3", "precision"),
        ("Actual-result precision", "actual_precision"),
        ("Recall@3", "recall"),
        ("Hit Rate@3", "hit_rate"),
        ("MRR", "mrr"),
    ]

    for metric_name, metric_key in metrics:

        dense = benchmark_results["Dense"][metric_key]
        bm25 = benchmark_results["BM25"][metric_key]
        hybrid = benchmark_results["Hybrid"][metric_key]

        print(
            f"{metric_name:<28} "
            f"{dense:>10.4f} "
            f"{bm25:>10.4f} "
            f"{hybrid:>10.4f}"
        )

    print("\nEvaluation questions:", len(dataset))
    print("Top K:", K)


if __name__ == "__main__":
    run_evaluation()
