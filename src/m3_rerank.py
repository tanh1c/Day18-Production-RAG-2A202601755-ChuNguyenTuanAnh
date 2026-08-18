from __future__ import annotations

"""Module 3: Cross-encoder reranking with latency benchmarking."""

import os
import sys
import time
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RERANK_TOP_K  # noqa: E402

_MODEL_CACHE: dict[str, object] = {}


@dataclass
class RerankResult:
    text: str
    original_score: float
    rerank_score: float
    metadata: dict
    rank: int


class CrossEncoderReranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3") -> None:
        self.model_name = model_name
        self._model = None

    def _load_model(self):
        """Lazy-load and process-cache the requested sentence-transformers model."""
        if self._model is None:
            if self.model_name not in _MODEL_CACHE:
                from sentence_transformers import CrossEncoder

                _MODEL_CACHE[self.model_name] = CrossEncoder(self.model_name)
            self._model = _MODEL_CACHE[self.model_name]
        return self._model

    def rerank(
        self,
        query: str,
        documents: list[dict],
        top_k: int = RERANK_TOP_K,
    ) -> list[RerankResult]:
        """Score query-document pairs and return the highest scoring documents."""
        if not documents or top_k <= 0:
            return []

        model = self._load_model()
        pairs = [(query, document["text"]) for document in documents]
        raw_scores = model.predict(pairs)

        try:
            scores = list(raw_scores)
        except TypeError:
            scores = [raw_scores]

        if len(scores) != len(documents):
            raise ValueError(
                "CrossEncoder returned a score count that does not match documents: "
                f"{len(scores)} != {len(documents)}"
            )

        scored = sorted(
            zip(scores, documents, strict=False),
            key=lambda item: float(item[0]),
            reverse=True,
        )

        return [
            RerankResult(
                text=document["text"],
                original_score=float(document.get("score", 0.0)),
                rerank_score=float(score),
                metadata=dict(document.get("metadata", {})),
                rank=rank,
            )
            for rank, (score, document) in enumerate(scored[:top_k], start=1)
        ]


class FlashrankReranker:
    """Optional lightweight reranker used only when explicitly selected."""

    def __init__(self) -> None:
        self._model = None

    def rerank(
        self,
        query: str,
        documents: list[dict],
        top_k: int = RERANK_TOP_K,
    ) -> list[RerankResult]:
        if not documents or top_k <= 0:
            return []

        from flashrank import Ranker, RerankRequest

        if self._model is None:
            self._model = Ranker()
        passages = [
            {"id": index, "text": document["text"]}
            for index, document in enumerate(documents)
        ]
        response = self._model.rerank(RerankRequest(query=query, passages=passages))
        results: list[RerankResult] = []
        for rank, item in enumerate(response[:top_k], start=1):
            original = documents[int(item.get("id", rank - 1))]
            results.append(
                RerankResult(
                    text=item["text"],
                    original_score=float(original.get("score", 0.0)),
                    rerank_score=float(item.get("score", 0.0)),
                    metadata=dict(original.get("metadata", {})),
                    rank=rank,
                )
            )
        return results


def benchmark_reranker(
    reranker,
    query: str,
    documents: list[dict],
    n_runs: int = 5,
) -> dict:
    """Benchmark reranking latency over repeated runs."""
    if n_runs <= 0:
        raise ValueError("n_runs must be positive")

    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        reranker.rerank(query, documents)
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)
    return {
        "avg_ms": sum(times) / len(times),
        "min_ms": min(times),
        "max_ms": max(times),
    }


if __name__ == "__main__":
    query = "Nhân viên được nghỉ phép bao nhiêu ngày?"
    docs = [
        {"text": "Nhân viên được nghỉ 12 ngày/năm.", "score": 0.8, "metadata": {}},
        {"text": "Mật khẩu thay đổi mỗi 90 ngày.", "score": 0.7, "metadata": {}},
        {"text": "Thời gian thử việc là 60 ngày.", "score": 0.75, "metadata": {}},
    ]
    reranker = CrossEncoderReranker()
    for result in reranker.rerank(query, docs):
        print(f"[{result.rank}] {result.rerank_score:.4f} | {result.text}")
