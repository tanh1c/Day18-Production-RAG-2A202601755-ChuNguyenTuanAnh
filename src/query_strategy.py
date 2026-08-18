from __future__ import annotations

"""Query decomposition and deterministic numeric guardrails for the RAG pipeline."""

import re

_QUESTION_CUES = (
    "bao nhiêu",
    "nào",
    "ai",
    "gì",
    "không",
    "mấy",
    "thế nào",
    "như thế nào",
    "bao lâu",
)


def _has_question_cue(text: str) -> bool:
    lowered = text.lower()
    return any(cue in lowered for cue in _QUESTION_CUES)


def _clean_fragment(text: str) -> str:
    fragment = re.sub(r"\s+", " ", text).strip(" .?")
    return f"{fragment}?" if fragment else ""


def query_variants(query: str) -> list[str]:
    """Return original query plus focused variants when multiple intents are present."""
    original = re.sub(r"\s+", " ", query).strip()
    if not original:
        return []

    focused: list[str] = []
    sentence_parts = [_clean_fragment(part) for part in re.split(r"\?\s*", original)]
    sentence_parts = [part for part in sentence_parts if part]
    if len(sentence_parts) > 1 and all(_has_question_cue(part) for part in sentence_parts):
        focused.extend(sentence_parts)
    else:
        conjunction_parts = re.split(
            r"\s+và\s+", original, maxsplit=1, flags=re.IGNORECASE
        )
        if (
            len(conjunction_parts) == 2
            and _has_question_cue(conjunction_parts[0])
            and _has_question_cue(conjunction_parts[1])
        ):
            focused.extend(_clean_fragment(part) for part in conjunction_parts)

    variants = [original]
    for candidate in focused:
        if candidate and candidate not in variants:
            variants.append(candidate)
    return variants


def _strip_markdown(text: str) -> str:
    return re.sub(r"[*_`#>]", "", text)


def _parse_amount_vnd(query: str) -> float | None:
    match = re.search(
        r"(\d+(?:[.,]\d+)?)\s*(triệu|tr\b|million|tỷ|billion)?",
        query.lower(),
    )
    if not match:
        return None
    value = float(match.group(1).replace(",", "."))
    unit = (match.group(2) or "").strip()
    if unit in {"triệu", "tr", "million"}:
        value *= 1_000_000
    elif unit in {"tỷ", "billion"}:
        value *= 1_000_000_000
    return value


def _format_vnd(value: float) -> str:
    return f"{int(round(value)):,}".replace(",", ".")


def monthly_pro_rata_fee_answer(query: str, contexts: list[str]) -> str | None:
    """Compute a monthly-rate overdue fee when all policy inputs are explicit."""
    joined = _strip_markdown("\n".join(contexts))
    lowered_query = query.lower()
    lowered_context = joined.lower()

    if "tạm ứng" not in lowered_query or "tạm ứng" not in lowered_context:
        return None

    amount = _parse_amount_vnd(query)
    elapsed_match = re.search(r"(?:sau\s+)?(\d+)\s*ngày", lowered_query)
    grace_match = re.search(r"trong vòng\s+(\d+)\s*ngày", lowered_context)
    rate_match = re.search(r"(\d+(?:[.,]\d+)?)\s*%\s*/?\s*tháng", lowered_context)

    if not (amount and elapsed_match and grace_match and rate_match):
        return None

    elapsed_days = int(elapsed_match.group(1))
    grace_days = int(grace_match.group(1))
    overdue_days = max(0, elapsed_days - grace_days)
    if overdue_days == 0:
        return "Khoản tạm ứng chưa quá hạn nên phí phạt là 0 VNĐ."

    monthly_rate = float(rate_match.group(1).replace(",", ".")) / 100.0
    fee = amount * monthly_rate * overdue_days / 30.0
    monthly_fee = amount * monthly_rate

    return (
        f"Khoản tạm ứng quá hạn {overdue_days} ngày; phí phạt pro-rata là "
        f"{_format_vnd(fee)} VNĐ "
        f"({_format_vnd(amount)} × {monthly_rate * 100:g}% = "
        f"{_format_vnd(monthly_fee)} VNĐ/tháng; {overdue_days}/30 tháng)."
    )
