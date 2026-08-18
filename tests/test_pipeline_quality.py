"""Regression checks for production answer-generation policy."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.pipeline import _answer_system_prompt


def test_answer_prompt_enforces_concise_grounded_output():
    prompt = _answer_system_prompt()

    assert "tối đa 2 câu" in prompt
    assert "không viết 'Giải thích:'" in prompt
    assert "chỉ trả đúng những phần" in prompt
    assert "pro-rata" in prompt
    assert "30 ngày" in prompt
    assert "không đủ thông tin" in prompt
