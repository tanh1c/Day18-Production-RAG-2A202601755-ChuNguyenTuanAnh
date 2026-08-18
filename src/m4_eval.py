from __future__ import annotations

"""Module 4: RAGAS evaluation and diagnostic failure analysis."""

import json
import math
import os
import sys
from dataclasses import asdict, dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import OPENAI_API_KEY, TEST_SET_PATH  # noqa: E402

METRIC_NAMES = (
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
)


@dataclass
class EvalResult:
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float


def load_test_set(path: str = TEST_SET_PATH) -> list[dict]:
    """Load the JSON evaluation set."""
    with open(path, encoding="utf-8") as file_obj:
        return json.load(file_obj)


def _zero_results() -> dict:
    return {
        "faithfulness": 0.0,
        "answer_relevancy": 0.0,
        "context_precision": 0.0,
        "context_recall": 0.0,
        "per_question": [],
    }


def _safe_float(value) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return 0.0
    return parsed if math.isfinite(parsed) else 0.0


def evaluate_ragas(
    questions: list[str],
    answers: list[str],
    contexts: list[list[str]],
    ground_truths: list[str],
) -> dict:
    """Evaluate answers with four RAGAS metrics and return aggregate + per-question data."""
    lengths = {len(questions), len(answers), len(contexts), len(ground_truths)}
    if len(lengths) != 1:
        raise ValueError("questions, answers, contexts and ground_truths must have equal length")
    if not questions:
        return _zero_results()
    if not OPENAI_API_KEY:
        print("  ⚠️  OPENAI_API_KEY is not set; returning zero RAGAS fallback.")
        return _zero_results()

    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import (
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )

        dataset = Dataset.from_dict(
            {
                "question": questions,
                "answer": answers,
                "contexts": contexts,
                "ground_truth": ground_truths,
            }
        )
        result = evaluate(
            dataset,
            metrics=[
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall,
            ],
            raise_exceptions=False,
        )
        dataframe = result.to_pandas()

        per_question: list[EvalResult] = []
        for index, row in dataframe.iterrows():
            row_contexts = row.get("contexts", contexts[index])
            if not isinstance(row_contexts, list):
                row_contexts = list(row_contexts) if row_contexts is not None else []
            per_question.append(
                EvalResult(
                    question=str(row.get("question", questions[index])),
                    answer=str(row.get("answer", answers[index])),
                    contexts=[str(context) for context in row_contexts],
                    ground_truth=str(row.get("ground_truth", ground_truths[index])),
                    faithfulness=_safe_float(row.get("faithfulness")),
                    answer_relevancy=_safe_float(row.get("answer_relevancy")),
                    context_precision=_safe_float(row.get("context_precision")),
                    context_recall=_safe_float(row.get("context_recall")),
                )
            )

        aggregates = {}
        for metric in METRIC_NAMES:
            values = [getattr(item, metric) for item in per_question]
            aggregates[metric] = sum(values) / len(values) if values else 0.0
        return {**aggregates, "per_question": per_question}
    except Exception as exc:  # pragma: no cover - external API/version behavior
        print(f"  ⚠️  RAGAS evaluation failed: {type(exc).__name__}: {exc}")
        return _zero_results()


def failure_analysis(
    eval_results: list[EvalResult],
    bottom_n: int = 10,
) -> list[dict]:
    """Return the worst questions with Diagnostic Error Tree diagnoses and fixes."""
    if bottom_n <= 0 or not eval_results:
        return []

    diagnostic_tree = {
        "faithfulness": (
            "Generation failure: answer contains claims insufficiently supported by retrieved context.",
            "Tighten the grounded-answer prompt, reduce generation freedom, and ensure supporting context is included.",
        ),
        "context_recall": (
            "Retrieval recall failure: one or more ground-truth facts are missing from retrieved chunks.",
            "Improve chunking, hybrid retrieval, parent expansion, or increase candidate recall before reranking.",
        ),
        "context_precision": (
            "Retrieval precision failure: irrelevant chunks remain in the final context.",
            "Improve cross-encoder reranking, metadata/version filtering, or reduce final context count.",
        ),
        "answer_relevancy": (
            "Generation relevance failure: the answer does not directly address the user question.",
            "Use a more explicit answer format and instruct the model to answer the requested quantity/decision first.",
        ),
    }

    analyzed = []
    for result in eval_results:
        metric_scores = {metric: _safe_float(getattr(result, metric)) for metric in METRIC_NAMES}
        average = sum(metric_scores.values()) / len(metric_scores)
        worst_metric = min(metric_scores, key=metric_scores.get)
        diagnosis, suggested_fix = diagnostic_tree[worst_metric]
        analyzed.append(
            {
                "question": result.question,
                "average_score": round(average, 6),
                "worst_metric": worst_metric,
                "score": round(metric_scores[worst_metric], 6),
                "metrics": {key: round(value, 6) for key, value in metric_scores.items()},
                "diagnosis": diagnosis,
                "suggested_fix": suggested_fix,
            }
        )

    analyzed.sort(key=lambda item: (item["average_score"], item["score"], item["question"]))
    return analyzed[:bottom_n]


def save_report(
    results: dict,
    failures: list[dict],
    path: str = "reports/ragas_report.json",
) -> None:
    """Persist aggregate metrics, serializable per-question results, and failures."""
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    per_question = results.get("per_question", [])
    report = {
        "aggregate": {
            metric: _safe_float(results.get(metric, 0.0)) for metric in METRIC_NAMES
        },
        "num_questions": len(per_question),
        "per_question": [
            asdict(item) if isinstance(item, EvalResult) else item for item in per_question
        ],
        "failures": failures,
    }
    with open(path, "w", encoding="utf-8") as file_obj:
        json.dump(report, file_obj, ensure_ascii=False, indent=2)
    print(f"Report saved to {path}")


if __name__ == "__main__":
    test_set = load_test_set()
    print(f"Loaded {len(test_set)} test questions")
    print("Run pipeline.py first to generate answers, then call evaluate_ragas().")
