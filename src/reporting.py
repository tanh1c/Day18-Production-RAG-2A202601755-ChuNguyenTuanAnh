from __future__ import annotations

"""Reporting helpers shared by the Lab 18 production pipeline."""

import json
import os
from collections.abc import Mapping
from dataclasses import asdict, is_dataclass

METRICS = (
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
)


def save_latency_report(
    timings_ms: Mapping[str, float],
    path: str = "reports/latency_report.json",
) -> None:
    """Save machine-readable latency data and a Markdown breakdown table."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    normalized = {key: round(float(value), 3) for key, value in timings_ms.items()}
    with open(path, "w", encoding="utf-8") as file_obj:
        json.dump({"unit": "ms", "timings": normalized}, file_obj, ensure_ascii=False, indent=2)

    markdown_path = os.path.splitext(path)[0] + ".md"
    lines = [
        "# Latency Breakdown",
        "",
        "| Step | Time (ms) |",
        "|---|---:|",
    ]
    lines.extend(f"| {step} | {value:.3f} |" for step, value in normalized.items())
    with open(markdown_path, "w", encoding="utf-8") as file_obj:
        file_obj.write("\n".join(lines) + "\n")


def _record(item) -> dict:
    if isinstance(item, dict):
        return item
    if is_dataclass(item):
        return asdict(item)
    return {
        "question": str(getattr(item, "question", "")),
        "answer": str(getattr(item, "answer", "")),
        "ground_truth": str(getattr(item, "ground_truth", "")),
    }


def _baseline_aggregate(path: str = "reports/naive_baseline_report.json") -> dict:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, encoding="utf-8") as file_obj:
            data = json.load(file_obj)
        return data.get("aggregate", {})
    except (OSError, json.JSONDecodeError):
        return {}


def write_failure_analysis(
    failures: list[dict],
    results: dict | None = None,
    path: str = "analysis/failure_analysis.md",
) -> None:
    """Write score comparison, bottom-5 details, and Diagnostic Error Tree analysis."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    results = results or {}
    production = {metric: float(results.get(metric, 0.0)) for metric in METRICS}
    baseline = _baseline_aggregate()
    per_question = {
        record.get("question", ""): record
        for record in (_record(item) for item in results.get("per_question", []))
    }

    lines = [
        "# Failure Analysis — Lab 18: Production RAG",
        "",
        "Phân tích được sinh từ RAGAS thực tế. Ground truth chỉ dùng ở bước đánh giá, "
        "không được dùng để hard-code retrieval hoặc generation.",
        "",
        "## RAGAS Scores",
        "",
        "| Metric | Naive Baseline | Production | Δ |",
        "|---|---:|---:|---:|",
    ]
    for metric in METRICS:
        baseline_score = float(baseline.get(metric, 0.0))
        production_score = production[metric]
        lines.append(
            f"| {metric} | {baseline_score:.4f} | {production_score:.4f} "
            f"| {production_score - baseline_score:+.4f} |"
        )

    lines.extend(
        [
            "",
            "## Diagnostic Error Tree",
            "",
            "1. **Context Recall thấp** → retrieval bỏ sót fact → kiểm tra chunking, hybrid recall, parent expansion.",
            "2. **Context Precision thấp** → context nhiễu/xung đột version → kiểm tra reranking, version metadata, final top-k.",
            "3. **Faithfulness thấp** → generation thêm claim hoặc suy luận sai → rút gọn answer, siết grounded prompt, kiểm tra arithmetic.",
            "4. **Answer Relevancy thấp** → answer không đi thẳng vào câu hỏi → ép format ngắn và chỉ trả đúng phần được hỏi.",
            "",
            "## Bottom-5 Failures",
            "",
        ]
    )

    if not failures:
        lines.extend(
            [
                "Chưa có per-question RAGAS data. Chạy workflow **Full RAG Evaluation** "
                "với `OPENAI_API_KEY` để cập nhật bottom-5 thực tế.",
                "",
            ]
        )
    else:
        for index, failure in enumerate(failures[:5], start=1):
            question = str(failure.get("question", ""))
            record = per_question.get(question, {})
            metrics = failure.get("metrics", {})
            answer = str(record.get("answer", "")).replace("\n", " ").strip()
            ground_truth = str(record.get("ground_truth", "")).replace("\n", " ").strip()
            lines.extend(
                [
                    f"### #{index} — {question}",
                    "",
                    f"- **Expected:** {ground_truth}",
                    f"- **Got:** {answer}",
                    (
                        "- **Metrics:** "
                        f"faithfulness `{float(metrics.get('faithfulness', 0.0)):.4f}`, "
                        f"answer_relevancy `{float(metrics.get('answer_relevancy', 0.0)):.4f}`, "
                        f"context_precision `{float(metrics.get('context_precision', 0.0)):.4f}`, "
                        f"context_recall `{float(metrics.get('context_recall', 0.0)):.4f}`"
                    ),
                    f"- **Worst metric:** `{failure.get('worst_metric', '')}` = `{float(failure.get('score', 0.0)):.4f}`",
                    f"- **Error Tree:** Output sai/chưa đủ → kiểm tra `{failure.get('worst_metric', '')}` → {failure.get('diagnosis', '')}",
                    f"- **Root cause:** {failure.get('diagnosis', '')}",
                    f"- **Suggested fix:** {failure.get('suggested_fix', '')}",
                    "",
                ]
            )

        first = failures[0]
        first_question = str(first.get("question", ""))
        first_record = per_question.get(first_question, {})
        lines.extend(
            [
                "## Case Study",
                "",
                f"**Question:** {first_question}",
                "",
                f"**Expected:** {str(first_record.get('ground_truth', '')).replace(chr(10), ' ')}",
                "",
                f"**Got:** {str(first_record.get('answer', '')).replace(chr(10), ' ')}",
                "",
                "### Error Tree walkthrough",
                "",
                "1. **Output đúng?** → So sánh `Got` với `Expected` và metric thấp nhất.",
                f"2. **Context đúng?** → Worst metric hiện tại là `{first.get('worst_metric', '')}`; dùng metric này để xác định retrieval hay generation là nút lỗi chính.",
                f"3. **Root cause:** {first.get('diagnosis', '')}",
                f"4. **Fix ở bước:** {first.get('suggested_fix', '')}",
                "",
                "### Nếu có thêm 1 giờ",
                "",
                "- Chạy lại full RAGAS sau mỗi thay đổi và chỉ chấp nhận submission khi quality gate đạt.",
                "- Thêm regression test cho grounded concise answer format và numeric-policy prompt.",
                "- Nếu lỗi thuộc version conflict, ưu tiên metadata/version filtering thay vì hard-code tên tài liệu.",
                "",
            ]
        )

    with open(path, "w", encoding="utf-8") as file_obj:
        file_obj.write("\n".join(lines))
