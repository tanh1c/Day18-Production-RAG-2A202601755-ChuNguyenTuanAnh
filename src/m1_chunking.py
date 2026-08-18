from __future__ import annotations

"""
Module 1: Advanced Chunking Strategies
=======================================
Semantic, hierarchical, and structure-aware chunking for Production RAG.
"""

import glob
import os
import re
import sys
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (  # noqa: E402
    DATA_DIR,
    HIERARCHICAL_CHILD_SIZE,
    HIERARCHICAL_PARENT_SIZE,
    SEMANTIC_THRESHOLD,
)


@dataclass
class Chunk:
    text: str
    metadata: dict = field(default_factory=dict)
    parent_id: str | None = None


def _extract_pdf_text(path: str) -> str:
    """Extract a PDF text layer; scanned PDFs without text return an empty string."""
    from pypdf import PdfReader

    reader = PdfReader(path)
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(pages).strip()


def load_documents(data_dir: str = DATA_DIR) -> list[dict]:
    """Load Markdown files and text-based PDFs from a data directory."""
    docs = []
    for fp in sorted(glob.glob(os.path.join(data_dir, "*.md"))):
        with open(fp, encoding="utf-8") as file_obj:
            docs.append(
                {"text": file_obj.read(), "metadata": {"source": os.path.basename(fp)}}
            )

    for fp in sorted(glob.glob(os.path.join(data_dir, "*.pdf"))):
        text = _extract_pdf_text(fp)
        if text:
            docs.append({"text": text, "metadata": {"source": os.path.basename(fp)}})
        else:
            print(
                f"  ⚠️  Bỏ qua {os.path.basename(fp)}: PDF scan ảnh, "
                "không có text layer (cần OCR)."
            )

    return docs


def chunk_basic(
    text: str,
    chunk_size: int = 500,
    metadata: dict | None = None,
) -> list[Chunk]:
    """Baseline paragraph chunking kept for comparison."""
    metadata = metadata or {}
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    current = ""
    for para in paragraphs:
        if len(current) + len(para) > chunk_size and current:
            chunks.append(
                Chunk(
                    text=current.strip(),
                    metadata={**metadata, "chunk_index": len(chunks)},
                )
            )
            current = ""
        current += para + "\n\n"
    if current.strip():
        chunks.append(
            Chunk(
                text=current.strip(),
                metadata={**metadata, "chunk_index": len(chunks)},
            )
        )
    return chunks


def _split_sentences(text: str) -> list[str]:
    """Split prose while treating paragraph boundaries as sentence boundaries."""
    return [
        part.strip()
        for part in re.split(r"(?<=[.!?])\s+|\n{2,}", text.strip())
        if part.strip()
    ]


def _lexical_similarity(left: str, right: str) -> float:
    """Deterministic fallback similarity used only when an embedding model fails."""
    left_tokens = set(re.findall(r"\w+", left.lower(), flags=re.UNICODE))
    right_tokens = set(re.findall(r"\w+", right.lower(), flags=re.UNICODE))
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / max(len(left_tokens | right_tokens), 1)


def chunk_semantic(
    text: str,
    threshold: float = SEMANTIC_THRESHOLD,
    metadata: dict | None = None,
) -> list[Chunk]:
    """Group consecutive sentences according to cosine semantic similarity."""
    metadata = metadata or {}
    sentences = _split_sentences(text)
    if not sentences:
        return []
    if len(sentences) == 1:
        return [
            Chunk(
                sentences[0],
                {**metadata, "strategy": "semantic", "chunk_index": 0},
            )
        ]

    similarities: list[float] = []
    try:
        import numpy as np
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings = model.encode(sentences, normalize_embeddings=True)
        similarities = [
            float(np.dot(embeddings[index - 1], embeddings[index]))
            for index in range(1, len(sentences))
        ]
    except Exception as exc:  # pragma: no cover - network/model fallback
        print(f"  ⚠️  Semantic model unavailable, lexical fallback: {exc}")
        similarities = [
            _lexical_similarity(sentences[index - 1], sentences[index])
            for index in range(1, len(sentences))
        ]

    groups: list[list[str]] = [[sentences[0]]]
    for sentence, similarity in zip(sentences[1:], similarities, strict=False):
        if similarity < threshold:
            groups.append([sentence])
        else:
            groups[-1].append(sentence)

    return [
        Chunk(
            text=" ".join(group).strip(),
            metadata={
                **metadata,
                "strategy": "semantic",
                "chunk_index": index,
                "sentence_count": len(group),
            },
        )
        for index, group in enumerate(groups)
        if group
    ]


def _split_oversized_text(text: str, max_size: int) -> list[str]:
    """Split an oversized unit by words while avoiding empty fragments."""
    text = text.strip()
    if not text:
        return []
    if max_size <= 0 or len(text) <= max_size:
        return [text]

    words = text.split()
    pieces: list[str] = []
    current: list[str] = []
    current_len = 0

    for word in words:
        if len(word) > max_size:
            if current:
                pieces.append(" ".join(current))
                current = []
                current_len = 0
            pieces.extend(
                word[offset : offset + max_size]
                for offset in range(0, len(word), max_size)
            )
            continue

        projected = current_len + (1 if current else 0) + len(word)
        if projected > max_size and current:
            pieces.append(" ".join(current))
            current = [word]
            current_len = len(word)
        else:
            current.append(word)
            current_len = projected

    if current:
        pieces.append(" ".join(current))
    return [piece for piece in pieces if piece]


def _pack_units(units: list[str], max_size: int) -> list[str]:
    """Pack text units into chunks that approximately respect max_size."""
    normalized: list[str] = []
    for unit in units:
        normalized.extend(_split_oversized_text(unit, max_size))

    packed: list[str] = []
    current = ""
    for unit in normalized:
        separator = "\n\n" if current else ""
        if current and len(current) + len(separator) + len(unit) > max_size:
            packed.append(current.strip())
            current = unit
        else:
            current = f"{current}{separator}{unit}" if current else unit
    if current.strip():
        packed.append(current.strip())
    return packed


def chunk_hierarchical(
    text: str,
    parent_size: int = HIERARCHICAL_PARENT_SIZE,
    child_size: int = HIERARCHICAL_CHILD_SIZE,
    metadata: dict | None = None,
) -> tuple[list[Chunk], list[Chunk]]:
    """Create parent chunks for context and smaller child chunks for retrieval."""
    metadata = metadata or {}
    if not text.strip():
        return [], []

    paragraphs = [part.strip() for part in re.split(r"\n{2,}", text) if part.strip()]
    parent_texts = _pack_units(paragraphs or [text], parent_size)

    parents: list[Chunk] = []
    children: list[Chunk] = []

    for parent_index, parent_text in enumerate(parent_texts):
        parent_id = f"parent_{parent_index}"
        parents.append(
            Chunk(
                text=parent_text,
                metadata={
                    **metadata,
                    "strategy": "hierarchical",
                    "chunk_type": "parent",
                    "parent_id": parent_id,
                    "chunk_index": parent_index,
                },
            )
        )

        sentence_units = _split_sentences(parent_text)
        child_texts = _pack_units(sentence_units or [parent_text], child_size)
        for child_text in child_texts:
            children.append(
                Chunk(
                    text=child_text,
                    metadata={
                        **metadata,
                        "strategy": "hierarchical",
                        "chunk_type": "child",
                        "child_index": len(children),
                    },
                    parent_id=parent_id,
                )
            )

    return parents, children


def chunk_structure_aware(text: str, metadata: dict | None = None) -> list[Chunk]:
    """Chunk Markdown by headings while preserving heading text and section metadata."""
    metadata = metadata or {}
    if not text.strip():
        return []

    header_pattern = re.compile(r"^(#{1,3})\s+(.+?)\s*$", flags=re.MULTILINE)
    matches = list(header_pattern.finditer(text))

    if not matches:
        return [
            Chunk(
                text=text.strip(),
                metadata={
                    **metadata,
                    "strategy": "structure",
                    "section": "document",
                    "chunk_index": 0,
                },
            )
        ]

    chunks: list[Chunk] = []
    preamble = text[: matches[0].start()].strip()
    if preamble:
        chunks.append(
            Chunk(
                text=preamble,
                metadata={
                    **metadata,
                    "strategy": "structure",
                    "section": "preamble",
                    "chunk_index": len(chunks),
                },
            )
        )

    for index, match in enumerate(matches):
        section_end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        section_text = text[match.start() : section_end].strip()
        header_text = match.group(2).strip()
        if not section_text:
            continue
        chunks.append(
            Chunk(
                text=section_text,
                metadata={
                    **metadata,
                    "strategy": "structure",
                    "section": header_text,
                    "header_level": len(match.group(1)),
                    "chunk_index": len(chunks),
                },
            )
        )

    return chunks


def compare_strategies(documents: list[dict]) -> dict:
    """Run all chunking strategies and return simple descriptive statistics."""

    def _stats(chunk_list: list[Chunk]) -> dict:
        lengths = [len(chunk.text) for chunk in chunk_list]
        if not lengths:
            return {"count": 0, "avg_len": 0, "min_len": 0, "max_len": 0}
        return {
            "count": len(lengths),
            "avg_len": round(sum(lengths) / len(lengths)),
            "min_len": min(lengths),
            "max_len": max(lengths),
        }

    all_text = "\n\n".join(document["text"] for document in documents)
    meta = {"source": "all"}

    basic = chunk_basic(all_text, metadata=meta)
    semantic = chunk_semantic(all_text, metadata=meta)
    parents, children = chunk_hierarchical(all_text, metadata=meta)
    structure = chunk_structure_aware(all_text, metadata=meta)

    results = {
        "basic": _stats(basic),
        "semantic": _stats(semantic),
        "hierarchical": {**_stats(children), "parents": len(parents)},
        "structure": _stats(structure),
    }

    print(f"{'Strategy':<15} {'Chunks':>7} {'Avg':>5} {'Min':>5} {'Max':>5}")
    for name, stats in results.items():
        print(
            f"{name:<15} {stats['count']:>7} {stats['avg_len']:>5} "
            f"{stats['min_len']:>5} {stats['max_len']:>5}"
        )

    return results


if __name__ == "__main__":
    docs = load_documents()
    print(f"Loaded {len(docs)} documents")
    strategy_results = compare_strategies(docs)
    for strategy_name, strategy_stats in strategy_results.items():
        print(f"  {strategy_name}: {strategy_stats}")
