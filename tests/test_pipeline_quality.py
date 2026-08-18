"""Regression checks for production answer-generation policy."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.pipeline import _answer_system_prompt


def test_single_intent_prompt_enforces_one_grounded_sentence():
    prompt = _answer_system_prompt(intent_count=1).lower()

    assert "một ý" in prompt
    assert "một câu hoàn chỉnh" in prompt
    assert "không viết tiêu đề 'giải thích:'" in prompt
    assert "không thêm quyền lợi" in prompt
    assert "không đủ thông tin" in prompt


def test_multi_intent_prompt_answers_each_intent_concisely():
    prompt = _answer_system_prompt(intent_count=2).lower()

    assert "câu hỏi có 2 ý" in prompt
    assert "tối đa 2 câu ngắn" in prompt
    assert "mỗi câu chỉ xử lý một ý" in prompt
    assert "không thêm fact phụ" in prompt
