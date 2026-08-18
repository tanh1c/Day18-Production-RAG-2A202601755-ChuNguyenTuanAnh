"""Regression tests for multi-intent retrieval and deterministic fee arithmetic."""

from src.query_strategy import monthly_pro_rata_fee_answer, query_variants


def test_query_variants_splits_two_intents():
    question = (
        "Một nhân viên Senior có 9 năm thâm niên được nghỉ bao nhiêu ngày phép năm "
        "và lương trong khoảng nào?"
    )
    variants = query_variants(question)
    assert variants[0] == question
    assert any("nghỉ bao nhiêu ngày phép năm" in item for item in variants[1:])
    assert any("lương trong khoảng nào" in item for item in variants[1:])


def test_query_variants_keeps_single_intent_intact():
    question = "Nhân viên thử việc có được hưởng bảo hiểm sức khỏe PVI không?"
    assert query_variants(question) == [question]


def test_query_variants_splits_two_question_sentences():
    question = (
        "Mentor và buddy có thể là cùng một người không? "
        "Quản lý trực tiếp có thể làm mentor không?"
    )
    variants = query_variants(question)
    assert len(variants) == 3
    assert "Mentor và buddy" in variants[1]
    assert "Quản lý trực tiếp" in variants[2]


def test_monthly_pro_rata_fee_answer():
    question = "Nhân viên tạm ứng 15 triệu, sau 20 ngày mới thanh toán. Bị phạt bao nhiêu?"
    context = (
        "Khoản tạm ứng phải được thanh toán trong vòng 15 ngày. "
        "Khoản tạm ứng chưa thanh toán sau 15 ngày sẽ bị tính phí 2%/tháng "
        "trên số tiền chưa hoàn ứng."
    )
    answer = monthly_pro_rata_fee_answer(question, [context])
    assert answer is not None
    assert "5 ngày" in answer
    assert "50.000 VNĐ" in answer
    assert "5/30" in answer


def test_monthly_pro_rata_fee_zero_when_not_overdue():
    question = "Nhân viên tạm ứng 15 triệu, sau 10 ngày thanh toán. Bị phạt bao nhiêu?"
    context = (
        "Khoản tạm ứng phải được thanh toán trong vòng 15 ngày. "
        "Khoản tạm ứng chưa thanh toán sau 15 ngày sẽ bị tính phí 2%/tháng "
        "trên số tiền chưa hoàn ứng."
    )
    assert monthly_pro_rata_fee_answer(question, [context]) == (
        "Khoản tạm ứng chưa quá hạn nên phí phạt là 0 VNĐ."
    )
