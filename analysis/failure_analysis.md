# Failure Analysis — Lab 18: Production RAG

Phân tích được sinh từ RAGAS thực tế. Ground truth chỉ dùng ở bước đánh giá, không được dùng để hard-code retrieval hoặc generation.

## RAGAS Scores

| Metric | Naive Baseline | Production | Δ |
|---|---:|---:|---:|
| faithfulness | 0.7903 | 0.8125 | +0.0222 |
| answer_relevancy | 0.7206 | 0.8047 | +0.0842 |
| context_precision | 0.9250 | 0.9500 | +0.0250 |
| context_recall | 0.9250 | 0.9500 | +0.0250 |

## Diagnostic Error Tree

1. **Context Recall thấp** → retrieval bỏ sót fact → kiểm tra chunking, hybrid recall, parent expansion.
2. **Context Precision thấp** → context nhiễu/xung đột version → kiểm tra reranking, version metadata, final top-k.
3. **Faithfulness thấp** → generation thêm claim hoặc suy luận sai → rút gọn answer, siết grounded prompt, kiểm tra arithmetic.
4. **Answer Relevancy thấp** → answer không đi thẳng vào câu hỏi → ép format ngắn và chỉ trả đúng phần được hỏi.

## Bottom-5 Failures

### #1 — Một nhân viên Senior có 9 năm thâm niên được nghỉ bao nhiêu ngày phép năm và lương trong khoảng nào?

- **Expected:** Theo chính sách v2024: 15 ngày cơ bản + 3 ngày thâm niên (9÷3=3) = 18 ngày phép. Lương Senior (P3-P4): 20-35 triệu VNĐ/tháng.
- **Got:** Nhân viên có 9 năm thâm niên được nghỉ **18 ngày phép năm** (15 + 3). Lương không được đề cập trong tài liệu, nên không có thông tin về khoảng lương.
- **Metrics:** faithfulness `0.2500`, answer_relevancy `0.0000`, context_precision `1.0000`, context_recall `0.5000`
- **Worst metric:** `answer_relevancy` = `0.0000`
- **Error Tree:** Output sai/chưa đủ → kiểm tra `answer_relevancy` → Generation relevance failure: the answer does not directly address the user question.
- **Root cause:** Generation relevance failure: the answer does not directly address the user question.
- **Suggested fix:** Use a more explicit answer format and instruct the model to answer the requested quantity/decision first.

### #2 — Nhân viên tạm ứng 15 triệu, sau 20 ngày mới thanh toán. Bị phạt bao nhiêu?

- **Expected:** Thời hạn thanh toán là 15 ngày. Quá hạn 5 ngày, bị tính phí 2%/tháng trên 15.000.000 VNĐ = 300.000 VNĐ/tháng (tính pro-rata khoảng 50.000 VNĐ cho 5 ngày).
- **Got:** Nhân viên tạm ứng 15 triệu VNĐ, sau 20 ngày mới thanh toán sẽ bị phạt 2%/tháng. Số tiền phạt là 2% của 15 triệu VNĐ, tương đương 300.000 VNĐ.
- **Metrics:** faithfulness `0.0000`, answer_relevancy `0.8945`, context_precision `1.0000`, context_recall `1.0000`
- **Worst metric:** `faithfulness` = `0.0000`
- **Error Tree:** Output sai/chưa đủ → kiểm tra `faithfulness` → Generation failure: answer contains claims insufficiently supported by retrieved context.
- **Root cause:** Generation failure: answer contains claims insufficiently supported by retrieved context.
- **Suggested fix:** Tighten the grounded-answer prompt, reduce generation freedom, and ensure supporting context is included.

### #3 — Thâm niên bao nhiêu năm thì được cộng thêm ngày phép?

- **Expected:** Theo chính sách v2024 hiện hành, nhân viên có thâm niên từ 3 năm trở lên được cộng thêm 1 ngày phép cho mỗi 3 năm. Chính sách cũ v2023 yêu cầu 5 năm.
- **Got:** Nhân viên có thâm niên từ **3 năm trở lên** được cộng thêm **1 ngày phép** cho mỗi 3 năm làm việc liên tục.
- **Metrics:** faithfulness `1.0000`, answer_relevancy `0.7921`, context_precision `0.5000`, context_recall `1.0000`
- **Worst metric:** `context_precision` = `0.5000`
- **Error Tree:** Output sai/chưa đủ → kiểm tra `context_precision` → Retrieval precision failure: irrelevant chunks remain in the final context.
- **Root cause:** Retrieval precision failure: irrelevant chunks remain in the final context.
- **Suggested fix:** Improve cross-encoder reranking, metadata/version filtering, or reduce final context count.

### #4 — Nhân viên được tài trợ khóa học 25 triệu, nghỉ việc sau 8 tháng hoàn thành khóa học. Phải hoàn trả bao nhiêu?

- **Expected:** Nhân viên phải cam kết làm việc ít nhất 1 năm sau khi hoàn thành khóa học. Nghỉ sau 8 tháng là trước hạn cam kết, phải hoàn trả 100% chi phí tức 25.000.000 VNĐ.
- **Got:** Nhân viên phải hoàn trả **100% chi phí** đào tạo đã được tài trợ, tức là **25.000.000 VNĐ**.
- **Metrics:** faithfulness `0.5000`, answer_relevancy `0.7986`, context_precision `1.0000`, context_recall `1.0000`
- **Worst metric:** `faithfulness` = `0.5000`
- **Error Tree:** Output sai/chưa đủ → kiểm tra `faithfulness` → Generation failure: answer contains claims insufficiently supported by retrieved context.
- **Root cause:** Generation failure: answer contains claims insufficiently supported by retrieved context.
- **Suggested fix:** Tighten the grounded-answer prompt, reduce generation freedom, and ensure supporting context is included.

### #5 — Nhân viên thử việc có được hưởng bảo hiểm sức khỏe PVI không?

- **Expected:** KHÔNG. Nhân viên thử việc chưa được hưởng gói bảo hiểm sức khỏe PVI. Chỉ được tham gia bảo hiểm xã hội bắt buộc.
- **Got:** Nhân viên thử việc chưa được hưởng gói bảo hiểm sức khỏe PVI. Họ chỉ được tham gia bảo hiểm xã hội bắt buộc.
- **Metrics:** faithfulness `0.5000`, answer_relevancy `0.8085`, context_precision `1.0000`, context_recall `1.0000`
- **Worst metric:** `faithfulness` = `0.5000`
- **Error Tree:** Output sai/chưa đủ → kiểm tra `faithfulness` → Generation failure: answer contains claims insufficiently supported by retrieved context.
- **Root cause:** Generation failure: answer contains claims insufficiently supported by retrieved context.
- **Suggested fix:** Tighten the grounded-answer prompt, reduce generation freedom, and ensure supporting context is included.

## Case Study

**Question:** Một nhân viên Senior có 9 năm thâm niên được nghỉ bao nhiêu ngày phép năm và lương trong khoảng nào?

**Expected:** Theo chính sách v2024: 15 ngày cơ bản + 3 ngày thâm niên (9÷3=3) = 18 ngày phép. Lương Senior (P3-P4): 20-35 triệu VNĐ/tháng.

**Got:** Nhân viên có 9 năm thâm niên được nghỉ **18 ngày phép năm** (15 + 3). Lương không được đề cập trong tài liệu, nên không có thông tin về khoảng lương.

### Error Tree walkthrough

1. **Output đúng?** → So sánh `Got` với `Expected` và metric thấp nhất.
2. **Context đúng?** → Worst metric hiện tại là `answer_relevancy`; dùng metric này để xác định retrieval hay generation là nút lỗi chính.
3. **Root cause:** Generation relevance failure: the answer does not directly address the user question.
4. **Fix ở bước:** Use a more explicit answer format and instruct the model to answer the requested quantity/decision first.

### Nếu có thêm 1 giờ

- Chạy lại full RAGAS sau mỗi thay đổi và chỉ chấp nhận submission khi quality gate đạt.
- Thêm regression test cho grounded concise answer format và numeric-policy prompt.
- Nếu lỗi thuộc version conflict, ưu tiên metadata/version filtering thay vì hard-code tên tài liệu.
