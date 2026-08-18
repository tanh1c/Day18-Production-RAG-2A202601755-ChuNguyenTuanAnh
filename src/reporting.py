from __future__ import annotations

"""Reporting helpers shared by the Lab 18 production pipeline."""

import json
import os
from collections.abc import Mapping


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


def write_failure_analysis(
    failures: list[dict],
    path: str = "analysis/failure_analysis.md",
) -> None:
    """Write the real bottom failures and Diagnostic Error Tree interpretation."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    lines = [
        "# Failure Analysis — Production RAG",
        "",
        "Phân tích này được sinh từ `reports/ragas_report.json`; không dùng ground truth để hard-code pipeline.",
        "",
        "## Diagnostic Error Tree",
        "",
        "1. **Context Recall thấp** → retrieval bỏ sót fact → xem chunking, hybrid recall, parent expansion.",
        "2. **Context Precision thấp** → context nhiễu → xem reranking, version metadata, final top-k.",
        "3. **Faithfulness thấp** → generation thêm claim ngoài context → siết grounded prompt.",
        "4. **Answer Relevancy thấp** → answer không đi thẳng vào câu hỏi → cải thiện answer format/prompt.",
        "",
        "## Bottom-5",
        "",
    ]

    if not failures:
        lines.extend(
            [
                "Chưa có per-question RAGAS data. Chạy workflow **Full RAG Evaluation** với `OPENAI_API_KEY` để cập nhật bottom-5 thực tế.",
                "",
            ]
        )
    else:
        for index, failure in enumerate(failures[:5], start=1):
            lines.extend(
                [
                    f"### {index}. {failure.get('question', '')}",
                    "",
                    f"- Average score: `{float(failure.get('average_score', 0.0)):.4f}`",
                    f"- Worst metric: `{failure.get('worst_metric', '')}` = `{float(failure.get('score', 0.0)):.4f}`",
                    f"- Diagnosis: {failure.get('diagnosis', '')}",
                    f"- Suggested fix: {failure.get('suggested_fix', '')}",
                    "",
                ]
            )

    with open(path, "w", encoding="utf-8") as file_obj:
        file_obj.write("\n".join(lines))
