"""
Basic RAG Baseline — run before Production RAG for an honest comparison.

Basic = paragraph chunking + dense-only retrieval, no reranking, no enrichment.
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import NAIVE_COLLECTION, OPENAI_API_KEY  # noqa: E402
from src.m1_chunking import chunk_basic, load_documents  # noqa: E402
from src.m2_search import DenseSearch  # noqa: E402
from src.m4_eval import evaluate_ragas, load_test_set, save_report  # noqa: E402


def main() -> dict:
    print("=" * 60)
    print("BASIC RAG BASELINE")
    print("(paragraph chunking + dense-only, no rerank, no enrichment)")
    print("=" * 60)

    docs = load_documents()
    chunks = []
    for doc in docs:
        for chunk in chunk_basic(doc["text"], metadata=doc["metadata"]):
            chunks.append({"text": chunk.text, "metadata": chunk.metadata})
    print(f"  {len(chunks)} basic paragraph chunks")

    search = DenseSearch()
    search.index(chunks, collection=NAIVE_COLLECTION)

    test_set = load_test_set()
    questions: list[str] = []
    answers: list[str] = []
    all_contexts: list[list[str]] = []
    ground_truths: list[str] = []

    llm_client = None
    if OPENAI_API_KEY:
        from openai import OpenAI

        llm_client = OpenAI(api_key=OPENAI_API_KEY)

    for index, item in enumerate(test_set):
        results = search.search(
            item["question"],
            top_k=3,
            collection=NAIVE_COLLECTION,
        )
        contexts = [result.text for result in results]

        if llm_client and contexts:
            try:
                context_str = "\n\n---\n\n".join(contexts)
                response = llm_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "Trả lời chỉ dựa trên context. Nếu context không đủ, "
                                "nói 'Không tìm thấy đủ thông tin trong tài liệu.'"
                            ),
                        },
                        {
                            "role": "user",
                            "content": (
                                f"Context:\n{context_str}\n\n"
                                f"Câu hỏi: {item['question']}"
                            ),
                        },
                    ],
                    temperature=0,
                    max_tokens=400,
                )
                answer = (response.choices[0].message.content or "").strip()
                if not answer:
                    answer = contexts[0]
            except Exception as exc:  # pragma: no cover - external API behavior
                print(f"  ⚠️  Baseline generation failed: {type(exc).__name__}: {exc}")
                answer = contexts[0]
        else:
            answer = contexts[0] if contexts else "Không tìm thấy."

        answers.append(answer)
        questions.append(item["question"])
        all_contexts.append(contexts)
        ground_truths.append(item["ground_truth"])
        print(f"  [{index + 1}/{len(test_set)}] {item['question'][:50]}...", flush=True)

    results = evaluate_ragas(questions, answers, all_contexts, ground_truths)
    print("\nBASIC BASELINE SCORES")
    for metric in (
        "faithfulness",
        "answer_relevancy",
        "context_precision",
        "context_recall",
    ):
        print(f"  {metric}: {float(results.get(metric, 0.0)):.4f}")
    save_report(results, [], path="reports/naive_baseline_report.json")
    print("\nDone! Now run the production pipeline with: python main.py")
    return results


if __name__ == "__main__":
    start = time.perf_counter()
    main()
    print(f"Total: {time.perf_counter() - start:.1f}s")
