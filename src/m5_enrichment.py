from __future__ import annotations

"""Module 5: pre-embedding chunk enrichment with deterministic fallbacks."""

import json
import os
import re
import sys
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import OPENAI_API_KEY  # noqa: E402

MODEL_NAME = "gpt-4o-mini"


@dataclass
class EnrichedChunk:
    original_text: str
    enriched_text: str
    summary: str
    hypothesis_questions: list[str]
    auto_metadata: dict
    method: str


def _client():
    from openai import OpenAI

    return OpenAI(api_key=OPENAI_API_KEY)


def _first_sentences(text: str, count: int = 2) -> str:
    sentences = [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+|\n+", text.strip())
        if sentence.strip()
    ]
    return " ".join(sentences[:count]).strip() if sentences else text.strip()


def _fallback_questions(text: str, n_questions: int = 3) -> list[str]:
    sentences = [
        sentence.strip().rstrip(".?!")
        for sentence in re.split(r"[.!?\n]+", text)
        if len(sentence.strip()) > 10
    ]
    questions = []
    for sentence in sentences[:n_questions]:
        questions.append(f"Thông tin nào được quy định về: {sentence}?")
    return questions


def _fallback_metadata(text: str) -> dict:
    lowered = text.lower()
    category_keywords = {
        "hr": ("nhân viên", "nghỉ", "lương", "thử việc", "mentor", "buddy"),
        "it": ("mật khẩu", "vpn", "mfa", "malware", "cntt", "dữ liệu"),
        "finance": ("chi phí", "tạm ứng", "mua sắm", "vnđ", "thanh toán"),
    }
    category = "policy"
    for candidate, keywords in category_keywords.items():
        if any(keyword in lowered for keyword in keywords):
            category = candidate
            break
    topic = _first_sentences(text, 1)[:120] or "general"
    return {
        "topic": topic,
        "entities": [],
        "category": category,
        "language": "vi",
    }


def _parse_json_object(content: str) -> dict:
    cleaned = content.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    parsed = json.loads(cleaned)
    if not isinstance(parsed, dict):
        raise ValueError("Expected an object from enrichment model")
    return parsed


def summarize_chunk(text: str) -> str:
    """Return a short Vietnamese summary, falling back to extractive summarization."""
    if OPENAI_API_KEY:
        try:
            response = _client().chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Tóm tắt đoạn văn sau trong tối đa 2 câu ngắn gọn bằng "
                            "tiếng Việt. Giữ nguyên số liệu và điều kiện quan trọng."
                        ),
                    },
                    {"role": "user", "content": text},
                ],
                temperature=0,
                max_tokens=150,
            )
            content = response.choices[0].message.content or ""
            if content.strip():
                return content.strip()
        except Exception as exc:  # pragma: no cover - external API behavior
            print(f"  ⚠️  OpenAI summarize failed: {type(exc).__name__}: {exc}")
    return _first_sentences(text, 2)


def generate_hypothesis_questions(text: str, n_questions: int = 3) -> list[str]:
    """Generate likely questions answered by a chunk (HyQA)."""
    if n_questions <= 0:
        return []
    if OPENAI_API_KEY:
        try:
            response = _client().chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            f"Tạo đúng {n_questions} câu hỏi tiếng Việt mà đoạn văn "
                            "có thể trả lời. Trả về mỗi câu trên một dòng, không giải thích."
                        ),
                    },
                    {"role": "user", "content": text},
                ],
                temperature=0,
                max_tokens=220,
            )
            content = response.choices[0].message.content or ""
            questions = []
            for line in content.splitlines():
                cleaned = re.sub(r"^\s*\d+[.)-]?\s*", "", line).strip(" -")
                if cleaned:
                    questions.append(cleaned if cleaned.endswith("?") else f"{cleaned}?")
            if questions:
                return questions[:n_questions]
        except Exception as exc:  # pragma: no cover - external API behavior
            print(f"  ⚠️  OpenAI HyQA failed: {type(exc).__name__}: {exc}")
    return _fallback_questions(text, n_questions)


def contextual_prepend(text: str, document_title: str = "") -> str:
    """Prepend a compact source/topic context while preserving the original text."""
    if OPENAI_API_KEY:
        try:
            response = _client().chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Viết đúng 1 câu ngắn mô tả đoạn văn thuộc tài liệu nào, "
                            "nói về chủ đề gì và nếu có thì trạng thái phiên bản/hiệu lực."
                        ),
                    },
                    {
                        "role": "user",
                        "content": f"Tài liệu: {document_title}\n\nĐoạn văn:\n{text}",
                    },
                ],
                temperature=0,
                max_tokens=100,
            )
            context = (response.choices[0].message.content or "").strip()
            if context:
                return f"{context}\n\n{text}"
        except Exception as exc:  # pragma: no cover - external API behavior
            print(f"  ⚠️  OpenAI contextual failed: {type(exc).__name__}: {exc}")

    source = document_title.strip() or "tài liệu nội bộ"
    summary = _first_sentences(text, 1)
    context = f"Ngữ cảnh nguồn: {source}; nội dung chính: {summary[:180]}"
    return f"{context}\n\n{text}"


def extract_metadata(text: str) -> dict:
    """Extract searchable metadata with an LLM or deterministic fallback."""
    if OPENAI_API_KEY:
        try:
            response = _client().chat.completions.create(
                model=MODEL_NAME,
                response_format={"type": "json_object"},
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Trích xuất metadata và chỉ trả JSON object với keys: "
                            '"topic", "entities" (array), "category" '
                            '(policy|hr|it|finance), "language" (vi|en).'
                        ),
                    },
                    {"role": "user", "content": text},
                ],
                temperature=0,
                max_tokens=180,
            )
            content = response.choices[0].message.content or "{}"
            parsed = _parse_json_object(content)
            if parsed:
                return parsed
        except Exception as exc:  # pragma: no cover - external API behavior
            print(f"  ⚠️  OpenAI metadata failed: {type(exc).__name__}: {exc}")
    return _fallback_metadata(text)


def _combined_fallback(text: str, source: str) -> dict:
    summary = _first_sentences(text, 2)
    source_label = source.strip() or "tài liệu nội bộ"
    return {
        "summary": summary,
        "questions": _fallback_questions(text, 3),
        "context": f"Trích từ {source_label}; đoạn này nói về {summary[:180]}",
        "metadata": _fallback_metadata(text),
    }


def _enrich_single_call(text: str, source: str) -> dict:
    """Get summary, HyQA, context and metadata in exactly one LLM call per chunk."""
    if OPENAI_API_KEY:
        try:
            response = _client().chat.completions.create(
                model=MODEL_NAME,
                response_format={"type": "json_object"},
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Phân tích đoạn văn và chỉ trả JSON object với schema: "
                            '{"summary":"tóm tắt 1-2 câu",'
                            '"questions":["q1","q2","q3"],'
                            '"context":"1 câu nêu nguồn, chủ đề và trạng thái phiên bản nếu có",'
                            '"metadata":{"topic":"...","entities":[],'
                            '"category":"policy|hr|it|finance","language":"vi|en"}}. '
                            "Giữ chính xác số liệu, phủ định, ngày hiệu lực và thông tin "
                            "văn bản thay thế nếu chúng xuất hiện."
                        ),
                    },
                    {
                        "role": "user",
                        "content": f"Tài liệu: {source}\n\nĐoạn văn:\n{text}",
                    },
                ],
                temperature=0,
                max_tokens=420,
            )
            content = response.choices[0].message.content or "{}"
            parsed = _parse_json_object(content)
            if all(key in parsed for key in ("summary", "questions", "context", "metadata")):
                return parsed
        except Exception as exc:  # pragma: no cover - external API behavior
            print(f"  ⚠️  Enrichment API failed: {type(exc).__name__}: {exc}")
    return _combined_fallback(text, source)


def enrich_chunks(
    chunks: list[dict],
    methods: list[str] | None = None,
) -> list[EnrichedChunk]:
    """Enrich chunks using combined mode by default or selected individual methods."""
    methods = list(methods) if methods is not None else ["combined"]
    allowed = {"summary", "hyqa", "contextual", "metadata", "combined"}
    unknown = set(methods) - allowed
    if unknown:
        raise ValueError(f"Unknown enrichment methods: {sorted(unknown)}")

    use_combined = "combined" in methods
    enriched: list[EnrichedChunk] = []

    for index, chunk in enumerate(chunks):
        text = str(chunk["text"])
        base_metadata = dict(chunk.get("metadata", {}))
        source = str(base_metadata.get("source", ""))

        if use_combined:
            result = _enrich_single_call(text, source)
            summary = str(result.get("summary", ""))
            questions = [str(item) for item in result.get("questions", []) if item]
            context_line = str(result.get("context", "")).strip()
            enriched_text = f"{context_line}\n\n{text}" if context_line else text
            model_metadata = result.get("metadata", {})
            auto_meta = model_metadata if isinstance(model_metadata, dict) else {}
        else:
            summary = summarize_chunk(text) if "summary" in methods else ""
            questions = (
                generate_hypothesis_questions(text) if "hyqa" in methods else []
            )
            enriched_text = (
                contextual_prepend(text, source) if "contextual" in methods else text
            )
            auto_meta = extract_metadata(text) if "metadata" in methods else {}

        enriched.append(
            EnrichedChunk(
                original_text=text,
                enriched_text=enriched_text,
                summary=summary,
                hypothesis_questions=questions,
                auto_metadata={**base_metadata, **auto_meta},
                method="+".join(methods),
            )
        )

        if (index + 1) % 10 == 0 or (index + 1) == len(chunks):
            print(f"  Enriched {index + 1}/{len(chunks)} chunks...", flush=True)

    return enriched


if __name__ == "__main__":
    sample = (
        "Nhân viên chính thức được nghỉ phép năm 12 ngày làm việc mỗi năm. "
        "Số ngày nghỉ phép tăng thêm 1 ngày cho mỗi 5 năm thâm niên công tác."
    )
    print("=== Enrichment Pipeline Demo ===\n")
    print(f"Original: {sample}\n")
    print(f"Summary: {summarize_chunk(sample)}\n")
    print(f"HyQA questions: {generate_hypothesis_questions(sample)}\n")
    print(f"Contextual: {contextual_prepend(sample, 'Sổ tay nhân viên')}\n")
    print(f"Auto metadata: {extract_metadata(sample)}")
