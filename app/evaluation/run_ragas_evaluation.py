
import asyncio
import json
from pathlib import Path

from app.orchestration.rag_graph import rag_app
from app.evaluation.ragas_evaluator import create_evaluator


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "rag_test_dataset.json"
)


def load_dataset():
    """Load evaluation questions and ground-truth answers."""
    with open(DATASET_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


def run_rag(question: str):
    """Execute one question through the LangGraph pipeline."""

    return rag_app.invoke({
        "question": question,
        "documents": [],
        "evidence_status": "",
        "faithfulness_status": "",
        "final_status": "",
        "answer": "",
        "sources": [],
    })


def extract_contexts(documents):
    """Extract text contexts from retrieved documents."""

    contexts = []

    for document in documents:
        if isinstance(document, dict):
            content = document.get("content", "")
        else:
            content = document.page_content

        if content and content.strip():
            contexts.append(content)

    return contexts


async def evaluate_question(test_case, metrics):
    """Evaluate one question using RAGAS and answerability."""

    question = test_case["question"]
    reference = test_case["ground_truth"]
    expected_answerable = test_case["answerable"]

    print("\n" + "=" * 60)
    print(f"Test case: {test_case['id']}")
    print(f"Question: {question}")

    # Run the RAG pipeline
    rag_result = run_rag(question)

    answer = rag_result["answer"]
    final_status = rag_result["final_status"]

    contexts = extract_contexts(
        rag_result.get("documents", [])
    )

    # Determine whether the system answered or abstained
    predicted_answerable = final_status == "answered"

    decision_correct = (
        expected_answerable == predicted_answerable
    )

    print(f"\nExpected answerable: {expected_answerable}")
    print(f"Predicted status: {final_status}")
    print(f"Decision correct: {decision_correct}")
    print(f"Generated answer: {answer}")
    print(f"Retrieved contexts: {len(contexts)}")

    # Only evaluate answer quality when an answer is expected
    scores = {}

    if expected_answerable:

        metric_inputs = {
            "faithfulness": {
                "user_input": question,
                "response": answer,
                "retrieved_contexts": contexts,
            },
            "answer_relevancy": {
                "user_input": question,
                "response": answer,
            },
            "answer_correctness": {
                "user_input": question,
                "response": answer,
                "reference": reference,
            },
            "context_precision": {
                "user_input": question,
                "retrieved_contexts": contexts,
                "reference": reference,
            },
            "context_recall": {
                "user_input": question,
                "retrieved_contexts": contexts,
                "reference": reference,
            },
        }

        for metric_name, metric in metrics.items():

            result = await metric.ascore(
                **metric_inputs[metric_name]
            )

            scores[metric_name] = result.value

            print(
                f"{metric_name}: {result.value:.4f}"
            )

    else:
        print(
            "\nRAGAS answer-quality evaluation skipped: "
            "this question is intentionally unanswerable."
        )

    return {
        "scores": scores,
        "expected_answerable": expected_answerable,
        "predicted_answerable": predicted_answerable,
        "decision_correct": decision_correct,
    }


def calculate_decision_metrics(results):
    """Calculate answerability and abstention metrics."""

    total = len(results)

    correct_decisions = sum(
        result["decision_correct"]
        for result in results
    )

    answerable_cases = [
        result for result in results
        if result["expected_answerable"]
    ]

    unanswerable_cases = [
        result for result in results
        if not result["expected_answerable"]
    ]

    false_abstentions = sum(
        not result["predicted_answerable"]
        for result in answerable_cases
    )

    unsupported_answers = sum(
        result["predicted_answerable"]
        for result in unanswerable_cases
    )

    answerability_accuracy = (
        correct_decisions / total if total else 0
    )

    false_abstention_rate = (
        false_abstentions / len(answerable_cases)
        if answerable_cases else 0
    )

    unsupported_answer_rate = (
        unsupported_answers / len(unanswerable_cases)
        if unanswerable_cases else 0
    )

    print("\n========== ANSWERABILITY EVALUATION ==========")

    print(
        f"Total test cases: {total}"
    )

    print(
        f"Correct decisions: {correct_decisions}/{total}"
    )

    print(
        f"Answerability accuracy: "
        f"{answerability_accuracy:.4f}"
    )

    print(
        f"False abstention rate: "
        f"{false_abstention_rate:.4f}"
    )

    print(
        f"Unsupported answer rate: "
        f"{unsupported_answer_rate:.4f}"
    )


def print_ragas_summary(all_results):
    """Calculate average RAGAS scores over answerable cases."""

    print("\n========== RAGAS ANSWER QUALITY RESULTS ==========")

    metric_names = [
        "faithfulness",
        "answer_relevancy",
        "answer_correctness",
        "context_precision",
        "context_recall",
    ]

    for metric_name in metric_names:

        values = [
            result["scores"][metric_name]
            for result in all_results
            if metric_name in result["scores"]
        ]

        if values:
            average = sum(values) / len(values)

            print(
                f"Mean {metric_name}: {average:.4f} "
                f"(based on {len(values)} answerable cases)"
            )


async def main():

    print("\n========== RAG EVALUATION ==========")

    dataset = load_dataset()
    metrics = create_evaluator()

    all_results = []

    for test_case in dataset:

        result = await evaluate_question(
            test_case,
            metrics,
        )

        all_results.append(result)

    print_ragas_summary(all_results)

    calculate_decision_metrics(all_results)


if __name__ == "__main__":
    asyncio.run(main())
