"""Lab 18 entry point: baseline -> production -> score comparison."""

import json
import os
import time

METRICS = (
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
)


def _load_report(path: str) -> dict:
    with open(path, encoding="utf-8") as file_obj:
        return json.load(file_obj)


def main() -> None:
    print("=" * 60)
    print("LAB 18: PRODUCTION RAG PIPELINE")
    print("=" * 60)
    start = time.perf_counter()
    os.makedirs("reports", exist_ok=True)

    print("\n📌 STEP 1: Running Basic RAG Baseline...")
    print("-" * 40)
    from naive_baseline import main as run_baseline

    run_baseline()

    print("\n📌 STEP 2: Running Production Pipeline...")
    print("-" * 40)
    from src.pipeline import build_pipeline, evaluate_pipeline

    search, reranker = build_pipeline()
    evaluate_pipeline(search, reranker)

    print("\n📌 STEP 3: Comparison")
    print("-" * 40)
    naive_path = "reports/naive_baseline_report.json"
    production_path = "reports/ragas_report.json"

    if os.path.exists(naive_path) and os.path.exists(production_path):
        naive = _load_report(naive_path)
        production = _load_report(production_path)

        print(f"\n{'Metric':<25} {'Basic':>8} {'Production':>12} {'Δ':>8}")
        print("-" * 55)
        for metric in METRICS:
            baseline_score = float(naive.get("aggregate", {}).get(metric, 0.0))
            production_score = float(
                production.get("aggregate", {}).get(metric, 0.0)
            )
            delta = production_score - baseline_score
            status = "✓" if production_score >= 0.75 else " "
            print(
                f"{status} {metric:<23} {baseline_score:>8.4f} "
                f"{production_score:>12.4f} {delta:>+8.4f}"
            )

    elapsed = time.perf_counter() - start
    print(f"\n⏱️  Total time: {elapsed:.1f}s")
    print("\n📋 Generated deliverables:")
    print("  - reports/naive_baseline_report.json")
    print("  - reports/ragas_report.json")
    print("  - reports/latency_report.json + .md")
    print("  - analysis/failure_analysis.md")
    print("\nRun final validation with: python check_lab.py")


if __name__ == "__main__":
    main()
