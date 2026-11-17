
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncOpenAI

from ragas.llms import llm_factory
from ragas.embeddings import HuggingFaceEmbeddings

from ragas.metrics.collections import (
    Faithfulness,
    AnswerRelevancy,
    AnswerCorrectness,
    ContextPrecision,
    ContextRecall,
)


# Load environment variables
PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")


def create_evaluator():
    """Initialize the RAGAS evaluation models."""

    client = AsyncOpenAI(
        api_key=os.environ["OPENROUTER_API_KEY"],
        base_url="https://openrouter.ai/api/v1",
    )

    evaluator_llm = llm_factory(
        "openai/gpt-4o-mini",
        provider="openai",
        client=client,
        max_tokens=500,
    )

    evaluator_embeddings = HuggingFaceEmbeddings(
        model="sentence-transformers/all-MiniLM-L6-v2",
        normalize_embeddings=True,
    )

    metrics = {
        "faithfulness": Faithfulness(llm=evaluator_llm),

        "answer_relevancy": AnswerRelevancy(
            llm=evaluator_llm,
            embeddings=evaluator_embeddings,
        ),

        "answer_correctness": AnswerCorrectness(
            llm=evaluator_llm,
            embeddings=evaluator_embeddings,
        ),

        "context_precision": ContextPrecision(
            llm=evaluator_llm,
        ),

        "context_recall": ContextRecall(
            llm=evaluator_llm,
        ),
    }

    return metrics
