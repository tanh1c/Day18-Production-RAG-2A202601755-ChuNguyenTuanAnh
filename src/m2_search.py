from __future__ import annotations

"""Module 2: Hybrid Search — Vietnamese BM25 + dense retrieval + RRF."""

import os
import sys
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (  # noqa: E402
    BM25_TOP_K,
    COLLECTION_NAME,
    DENSE_TOP_K,
    EMBEDDING_DIM,
    EMBEDDING_MODEL,
    HYBRID_TOP_K,
    QDRANT_HOST,
    QDRANT_PORT,
)


@dataclass
class SearchResult:
    text: str
    score: float
    metadata: dict
    method: str


def segment_vietnamese(text: str) -> str:
    """Segment Vietnamese words and normalize compound-word underscores."""
    try:
        from underthesea import word_tokenize

        segmented = word_tokenize(text, format="text")
        return segmented.replace("_", " ")
    except Exception as exc:  # pragma: no cover - dependency fallback
        print(f"  ⚠️  Vietnamese segmenter unavailable, whitespace fallback: {exc}")
        return text


class BM25Search:
    def __init__(self) -> None:
        self.corpus_tokens: list[list[str]] = []
        self.documents: list[dict] = []
        self.bm25 = None

    def index(self, chunks: list[dict]) -> None:
        """Build a BM25Okapi index over segmented chunk text."""
        from rank_bm25 import BM25Okapi

        self.documents = list(chunks)
        self.corpus_tokens = [
            segment_vietnamese(chunk["text"]).lower().split() for chunk in self.documents
        ]
        self.bm25 = BM25Okapi(self.corpus_tokens) if self.corpus_tokens else None

    def search(self, query: str, top_k: int = BM25_TOP_K) -> list[SearchResult]:
        """Return positive-scoring BM25 results ordered by relevance."""
        if self.bm25 is None or not self.documents:
            return []

        query_tokens = segment_vietnamese(query).lower().split()
        if not query_tokens:
            return []
        scores = self.bm25.get_scores(query_tokens)
        top_indices = sorted(
            range(len(scores)),
            key=lambda index: float(scores[index]),
            reverse=True,
        )[:top_k]

        results: list[SearchResult] = []
        for index in top_indices:
            score = float(scores[index])
            if score <= 0:
                continue
            document = self.documents[index]
            results.append(
                SearchResult(
                    text=document["text"],
                    score=score,
                    metadata=dict(document.get("metadata", {})),
                    method="bm25",
                )
            )
        return results


class DenseSearch:
    def __init__(self) -> None:
        from qdrant_client import QdrantClient

        self.client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        self._encoder = None

    def _get_encoder(self):
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer

            self._encoder = SentenceTransformer(EMBEDDING_MODEL)
        return self._encoder

    def index(self, chunks: list[dict], collection: str = COLLECTION_NAME) -> None:
        """Embed chunks with BGE-M3 and upsert them to a Qdrant collection."""
        from qdrant_client.models import Distance, PointStruct, VectorParams

        self.client.recreate_collection(
            collection_name=collection,
            vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
        )
        if not chunks:
            return

        texts = [chunk["text"] for chunk in chunks]
        vectors = self._get_encoder().encode(
            texts,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        points = []
        for index, (chunk, vector) in enumerate(zip(chunks, vectors, strict=False)):
            metadata = dict(chunk.get("metadata", {}))
            payload = {**metadata, "text": chunk["text"]}
            points.append(PointStruct(id=index, vector=vector.tolist(), payload=payload))
        self.client.upsert(collection_name=collection, points=points, wait=True)

    def search(
        self,
        query: str,
        top_k: int = DENSE_TOP_K,
        collection: str = COLLECTION_NAME,
    ) -> list[SearchResult]:
        """Query Qdrant with a dense query vector."""
        if not query.strip():
            return []
        query_vector = self._get_encoder().encode(
            query,
            normalize_embeddings=True,
        ).tolist()
        response = self.client.query_points(
            collection_name=collection,
            query=query_vector,
            limit=top_k,
            with_payload=True,
        )

        results: list[SearchResult] = []
        for point in response.points:
            payload = dict(point.payload or {})
            text = str(payload.pop("text", ""))
            if not text:
                continue
            results.append(
                SearchResult(
                    text=text,
                    score=float(point.score),
                    metadata=payload,
                    method="dense",
                )
            )
        return results


def reciprocal_rank_fusion(
    results_list: list[list[SearchResult]],
    k: int = 60,
    top_k: int = HYBRID_TOP_K,
) -> list[SearchResult]:
    """Merge ranked result lists with reciprocal-rank fusion."""
    fused: dict[str, dict] = {}

    for result_list in results_list:
        for rank, result in enumerate(result_list):
            entry = fused.setdefault(
                result.text,
                {"score": 0.0, "result": result, "best_rank": rank},
            )
            entry["score"] += 1.0 / (k + rank + 1)
            entry["best_rank"] = min(entry["best_rank"], rank)

    ordered = sorted(
        fused.values(),
        key=lambda entry: (-entry["score"], entry["best_rank"], entry["result"].text),
    )[:top_k]

    return [
        SearchResult(
            text=entry["result"].text,
            score=float(entry["score"]),
            metadata=dict(entry["result"].metadata),
            method="hybrid",
        )
        for entry in ordered
    ]


class HybridSearch:
    """Combine BM25 and dense retrieval with RRF."""

    def __init__(self) -> None:
        self.bm25 = BM25Search()
        self.dense = DenseSearch()

    def index(self, chunks: list[dict]) -> None:
        self.bm25.index(chunks)
        self.dense.index(chunks)

    def search(self, query: str, top_k: int = HYBRID_TOP_K) -> list[SearchResult]:
        bm25_results = self.bm25.search(query, top_k=BM25_TOP_K)
        dense_results = self.dense.search(query, top_k=DENSE_TOP_K)
        return reciprocal_rank_fusion([bm25_results, dense_results], top_k=top_k)


if __name__ == "__main__":
    sample = "Nhân viên được nghỉ phép năm"
    print(f"Original:  {sample}")
    print(f"Segmented: {segment_vietnamese(sample)}")
