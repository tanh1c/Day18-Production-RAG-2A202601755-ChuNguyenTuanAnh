from __future__ import annotations

"""Production RAG pipeline: M1 -> M5 -> M2 -> M3 -> answer -> M4."""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import OPENAI_API_KEY, RERANK_TOP_K  # noqa: E402
from src.m1_chunking import chunk_hierarchical, load_documents  # noqa: E402
from src.m2_search import HybridSearch  # noqa: E402
from src.m3_rerank import CrossEncoderReranker  # noqa: E402
from src.m4_eval import (  # noqa: E402
    evaluate_ragas,
    failure_analysis,
    load_test_set,
    save_report,
)
from src.m5_enrichment import enrich_chunks  # noqa: E402
from src.reporting import save_latency_report, write_failure_analysis  # noqa: E402


def _ms(start: float) -> float:
    return (time.perf_counter() - start) * 1000.0


def build_pipeline() -> tuple[HybridSearch, CrossEncoderReranker]:
    """Build and index the production retrieval pipeline."""
    print("=" * 60)
    print("PRODUCTION RAG PIPELINE")
    print("=" * 60, flush=True)

    build_timings: dict[str, float] = {}

    start = time.perf_counter()
    print("\n[1/4] Loading and hierarchical chunking...", flush=True)
    docs = load_documents()
    all_chunks: list[dict] = []
    parent_lookup: dict[str, str] = {}

    for doc in docs:
        source = str(doc.get("metadata", {}).get("source", "unknown"))
        parents, children = chunk_hierarchical(doc["text"], metadata=doc["metadata"])
        for parent in parents:
            parent_id = str(parent.metadata.get("parent_id", ""))
            parent_key = f"{source}::{parent_id}"
            parent_lookup[parent_key] = parent.text
        for child in children:
            parent_key = f"{source}::{child.parent_id}"
            all_chunks.append(
                {
                    "text": child.text,
                    "metadata": {
                        **child.metadata,
                        "parent_id": child.parent_id,
                        "parent_key": parent_key,
                        "source": source,
                    },
                }
            )
    build_timings["load_and_chunk"] = _ms(start)
    print(
        f"  ✓ {len(all_chunks)} child chunks / {len(parent_lookup)} parents "
        f"from {len(docs)} documents ({build_timings['load_and_chunk']:.1f}ms)",
        flush=True,
    )

    start = time.perf_counter()
    print(f"\n[2/4] Enriching {len(all_chunks)} chunks (combined mode)...", flush=True)
    enriched = enrich_chunks(all_chunks)
    if enriched:
        all_chunks = [
            {
                "text": item.enriched_text,
                "metadata": {
                    **item.auto_metadata,
                    "original_text": item.original_text,
                    "summary": item.summary,
                    "hypothesis_questions": item.hypothesis_questions,
                },
            }
            for item in enriched
        ]
    build_timings["enrichment"] = _ms(start)
    print(
        f"  ✓ Enriched {len(all_chunks)} chunks ({build_timings['enrichment']:.1f}ms)",
        flush=True,
    )

    start = time.perf_counter()
    print(f"\n[3/4] Indexing {len(all_chunks)} chunks (BM25 + Dense)...", flush=True)
    search = HybridSearch()
    search.index(all_chunks)
    build_timings["hybrid_indexing"] = _ms(start)
    print(f"  ✓ Indexed ({build_timings['hybrid_indexing']:.1f}ms)", flush=True)

    start = time.perf_counter()
    print("\n[4/4] Preparing cross-encoder reranker...", flush=True)
    reranker = CrossEncoderReranker()
    reranker._load_model()
    build_timings["reranker_load"] = _ms(start)
    print(f"  ✓ Reranker ready ({build_timings['reranker_load']:.1f}ms)", flush=True)

    search.parent_lookup = parent_lookup
    search.build_timings_ms = build_timings
    search.last_latency_ms = {}
    return search, reranker


def _expand_parent_contexts(reranked_results, search: HybridSearch) -> list[str]:
    parent_lookup = getattr(search, "parent_lookup", {})
    contexts: list[str] = []
    seen: set[str] = set()

    for result in reranked_results:
        parent_key = str(result.metadata.get("parent_key", ""))
        context = parent_lookup.get(parent_key)
        if not context:
            context = str(result.metadata.get("original_text", "")) or result.text
        normalized = context.strip()
        if normalized and normalized not in seen:
            seen.add(normalized)
            contexts.append(normalized)
    return contexts


def _answer_system_prompt() -> str:
    """Return the strict grounded-answer policy used by production generation."""
    return (
        "Bạn là trợ lý hỏi đáp chính sách nội bộ. Chỉ dùng thông tin có trong CONTEXT; "
        "không bịa, không thêm mục đích/lý do nếu câu hỏi không yêu cầu. "
        "QUY TẮC OUTPUT: trả lời tối đa 2 câu ngắn; không viết 'Giải thích:'; "
        "không lặp lại cùng một fact dưới nhiều cách diễn đạt; chỉ trả đúng những phần "
        "người dùng hỏi. Nếu câu hỏi có nhiều phần, trả đủ từng phần nhưng vẫn ngắn gọn. "
        "Nếu có nhiều phiên bản chính sách, ưu tiên văn bản ghi là hiện hành, có ngày hiệu "
        "lực mới hơn, hoặc ghi rõ thay thế phiên bản cũ; chỉ nhắc phiên bản cũ khi cần để "
        "giải quyết xung đột. Giữ chính xác phủ định, ngưỡng, đơn vị và số liệu. "
        "Với câu hỏi tính toán, chỉ dùng quy tắc/số liệu trong CONTEXT và tính cẩn thận. "
        "Nếu rate được cho theo tháng nhưng khoảng thời gian thực tế ngắn hơn một tháng và "
        "CONTEXT không quy định cách làm tròn khác, tính pro-rata theo số ngày thực tế trên "
        "chu kỳ 30 ngày. Có thể ghi một công thức ngắn, nhưng không thêm diễn giải ngoài "
        "phép tính cần thiết. Nếu CONTEXT không đủ cho một phần câu hỏi, nói rõ phần đó "
        "không đủ thông tin trong tài liệu thay vì suy đoán."
    )


def _generate_answer(query: str, contexts: list[str]) -> str:
    if not contexts:
        return "Không tìm thấy thông tin."
    if not OPENAI_API_KEY:
        return contexts[0]

    try:
        from openai import OpenAI

        client = OpenAI(api_key=OPENAI_API_KEY)
        context_str = "\n\n---\n\n".join(contexts)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": _answer_system_prompt()},
                {
                    "role": "user",
                    "content": f"CONTEXT:\n{context_str}\n\nCÂU HỎI: {query}",
                },
            ],
            temperature=0,
            max_tokens=180,
        )
        content = response.choices[0].message.content or ""
        return content.strip() or contexts[0]
    except Exception as exc:  # pragma: no cover - external API behavior
        print(f"  ⚠️  LLM generation failed: {type(exc).__name__}: {exc}", flush=True)
        return contexts[0]


def run_query(
    query: str,
    search: HybridSearch,
    reranker: CrossEncoderReranker,
) -> tuple[str, list[str]]:
    """Run retrieval, reranking, parent expansion, and grounded generation."""
    timings: dict[str, float] = {}

    start = time.perf_counter()
    results = search.search(query)
    timings["retrieval"] = _ms(start)

    documents = [
        {
            "text": result.text,
            "score": result.score,
            "metadata": result.metadata,
        }
        for result in results
    ]

    start = time.perf_counter()
    reranked = reranker.rerank(query, documents, top_k=RERANK_TOP_K)
    timings["reranking"] = _ms(start)

    contexts = _expand_parent_contexts(reranked, search)
    if not contexts:
        contexts = [
            str(result.metadata.get("original_text", "")) or result.text
            for result in results[:RERANK_TOP_K]
        ]

    start = time.perf_counter()
    answer = _generate_answer(query, contexts)
    timings["generation"] = _ms(start)
    timings["query_total"] = sum(timings.values())
    search.last_latency_ms = timings
    return answer, contexts


def evaluate_pipeline(
    search: HybridSearch,
    reranker: CrossEncoderReranker,
) -> dict:
    """Evaluate the production pipeline and write reports/analysis artifacts."""
    test_set = load_test_set()
    print(f"\n[Eval] Running {len(test_set)} queries...", flush=True)

    questions: list[str] = []
    answers: list[str] = []
    all_contexts: list[list[str]] = []
    ground_truths: list[str] = []
    query_timings: dict[str, list[float]] = {
        "retrieval": [],
        "reranking": [],
        "generation": [],
        "query_total": [],
    }

    evaluation_start = time.perf_counter()
    for index, item in enumerate(test_set):
        answer, contexts = run_query(item["question"], search, reranker)
        questions.append(item["question"])
        answers.append(answer)
        all_contexts.append(contexts)
        ground_truths.append(item["ground_truth"])
        last = getattr(search, "last_latency_ms", {})
        for key in query_timings:
            query_timings[key].append(float(last.get(key, 0.0)))
        print(f"  [{index + 1}/{len(test_set)}] {item['question'][:60]}...", flush=True)

    ragas_start = time.perf_counter()
    print(
        f"\n[Eval] Running RAGAS (4 metrics × {len(test_set)} questions)...",
        flush=True,
    )
    results = evaluate_ragas(questions, answers, all_contexts, ground_truths)
    ragas_ms = _ms(ragas_start)
    print(f"  ✓ RAGAS done ({ragas_ms:.1f}ms)", flush=True)

    print("\n" + "=" * 60)
    print("PRODUCTION RAG SCORES")
    print("=" * 60)
    for metric in (
        "faithfulness",
        "answer_relevancy",
        "context_precision",
        "context_recall",
    ):
        score = float(results.get(metric, 0.0))
        print(f"  {'✓' if score >= 0.75 else '✗'} {metric}: {score:.4f}")

    failures = failure_analysis(results.get("per_question", []), bottom_n=5)
    save_report(results, failures, path="reports/ragas_report.json")
    save_report(results, failures, path="ragas_report.json")
    write_failure_analysis(failures, results=results)

    latency: dict[str, float] = dict(getattr(search, "build_timings_ms", {}))
    for key, values in query_timings.items():
        latency[f"avg_{key}_per_question"] = sum(values) / len(values) if values else 0.0
    latency["ragas_evaluation"] = ragas_ms
    latency["production_evaluation_total"] = _ms(evaluation_start)
    save_latency_report(latency)
    return results


if __name__ == "__main__":
    total_start = time.perf_counter()
    hybrid_search, cross_encoder = build_pipeline()
    evaluate_pipeline(hybrid_search, cross_encoder)
    print(f"\nTotal: {_ms(total_start) / 1000.0:.1f}s")
