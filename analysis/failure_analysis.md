# Failure Analysis — Lab 18: Production RAG

Phân tích được sinh từ RAGAS thực tế. Ground truth chỉ dùng ở bước đánh giá, không được dùng để hard-code retrieval hoặc generation.

## RAGAS Scores

| Metric | Naive Baseline | Production | Δ |
|---|---:|---:|---:|
| faithfulness | 0.7803 | 0.8375 | +0.0572 |
| answer_relevancy | 0.7236 | 0.8355 | +0.1119 |
| context_precision | 0.9250 | 0.9417 | +0.0167 |
| context_recall | 0.9250 | 0.9750 | +0.0500 |

## Diagnostic Error Tree

1. **Context Recall thấp** → retrieval bỏ sót fact → kiểm tra chunking, hybrid recall, parent expansion.
2. **Context Precision thấp** → context nhiễu/xung đột version → kiểm tra reranking, version metadata, final top-k.
3. **Faithfulness thấp** → generation thêm claim hoặc suy luận sai → rút gọn answer, siết grounded prompt, kiểm tra arithmetic.
4. **Answer Relevancy thấp** → answer không đi thẳng vào câu hỏi → ép format ngắn và chỉ trả đúng phần được hỏi.

## Bottom-5 Failures

### #1 — Lương thử việc của nhân viên Junior mức cao nhất là bao nhiêu?

- **Expected:** Junior cao nhất là 20.000.000 VNĐ/tháng. Lương thử việc = 85% x 20.000.000 = 17.000.000 VNĐ/tháng.
- **Got:** Lương thử việc của nhân viên Junior mức cao nhất là 17.000.000 VNĐ/tháng.
- **Metrics:** faithfulness `0.0000`, answer_relevancy `0.8241`, context_precision `1.0000`, context_recall `1.0000`
- **Worst metric:** `faithfulness` = `0.0000`
- **Error Tree:** Output sai/chưa đủ → kiểm tra `faithfulness` → Generation failure: answer contains claims insufficiently supported by retrieved context.
- **Root cause:** Generation failure: answer contains claims insufficiently supported by retrieved context.
- **Suggested fix:** Tighten the grounded-answer prompt, reduce generation freedom, and ensure supporting context is included.

### #2 — Nhân viên tạm ứng 15 triệu, sau 20 ngày mới thanh toán. Bị phạt bao nhiêu?

- **Expected:** Thời hạn thanh toán là 15 ngày. Quá hạn 5 ngày, bị tính phí 2%/tháng trên 15.000.000 VNĐ = 300.000 VNĐ/tháng (tính pro-rata khoảng 50.000 VNĐ cho 5 ngày).
- **Got:** Khoản tạm ứng quá hạn 5 ngày; phí phạt pro-rata là 50.000 VNĐ (15.000.000 × 2% = 300.000 VNĐ/tháng; 5/30 tháng).
- **Metrics:** faithfulness `0.2500`, answer_relevancy `0.7872`, context_precision `1.0000`, context_recall `1.0000`
- **Worst metric:** `faithfulness` = `0.2500`
- **Error Tree:** Output sai/chưa đủ → kiểm tra `faithfulness` → Generation failure: answer contains claims insufficiently supported by retrieved context.
- **Root cause:** Generation failure: answer contains claims insufficiently supported by retrieved context.
- **Suggested fix:** Tighten the grounded-answer prompt, reduce generation freedom, and ensure supporting context is included.

### #3 — Thâm niên bao nhiêu năm thì được cộng thêm ngày phép?

- **Expected:** Theo chính sách v2024 hiện hành, nhân viên có thâm niên từ 3 năm trở lên được cộng thêm 1 ngày phép cho mỗi 3 năm. Chính sách cũ v2023 yêu cầu 5 năm.
- **Got:** Nhân viên có thâm niên từ **3 năm trở lên** được cộng thêm **1 ngày phép** cho mỗi 3 năm làm việc liên tục.
- **Metrics:** faithfulness `1.0000`, answer_relevancy `0.7902`, context_precision `0.5000`, context_recall `1.0000`
- **Worst metric:** `context_precision` = `0.5000`
- **Error Tree:** Output sai/chưa đủ → kiểm tra `context_precision` → Retrieval precision failure: irrelevant chunks remain in the final context.
- **Root cause:** Retrieval precision failure: irrelevant chunks remain in the final context.
- **Suggested fix:** Improve cross-encoder reranking, metadata/version filtering, or reduce final context count.

### #4 — Nhân viên được tài trợ khóa học 25 triệu, nghỉ việc sau 8 tháng hoàn thành khóa học. Phải hoàn trả bao nhiêu?

- **Expected:** Nhân viên phải cam kết làm việc ít nhất 1 năm sau khi hoàn thành khóa học. Nghỉ sau 8 tháng là trước hạn cam kết, phải hoàn trả 100% chi phí tức 25.000.000 VNĐ.
- **Got:** Nhân viên phải hoàn trả 100% chi phí đào tạo đã được tài trợ, tức là 25 triệu VNĐ.
- **Metrics:** faithfulness `0.5000`, answer_relevancy `0.7960`, context_precision `1.0000`, context_recall `1.0000`
- **Worst metric:** `faithfulness` = `0.5000`
- **Error Tree:** Output sai/chưa đủ → kiểm tra `faithfulness` → Generation failure: answer contains claims insufficiently supported by retrieved context.
- **Root cause:** Generation failure: answer contains claims insufficiently supported by retrieved context.
- **Suggested fix:** Tighten the grounded-answer prompt, reduce generation freedom, and ensure supporting context is included.

### #5 — Có cần kích hoạt xác thực đa yếu tố (MFA) không?

- **Expected:** Có, theo chính sách mật khẩu v2.0 hiện hành, tất cả nhân viên bắt buộc kích hoạt MFA cho email, VPN và hệ thống nội bộ. Chính sách cũ v1.0 không yêu cầu MFA.
- **Got:** Có, tất cả nhân viên bắt buộc phải kích hoạt MFA cho tài khoản email, VPN và các hệ thống nội bộ.
- **Metrics:** faithfulness `1.0000`, answer_relevancy `0.8079`, context_precision `1.0000`, context_recall `0.5000`
- **Worst metric:** `context_recall` = `0.5000`
- **Error Tree:** Output sai/chưa đủ → kiểm tra `context_recall` → Retrieval recall failure: one or more ground-truth facts are missing from retrieved chunks.
- **Root cause:** Retrieval recall failure: one or more ground-truth facts are missing from retrieved chunks.
- **Suggested fix:** Improve chunking, hybrid retrieval, parent expansion, or increase candidate recall before reranking.

## Case Study

**Question:** Lương thử việc của nhân viên Junior mức cao nhất là bao nhiêu?

**Expected:** Junior cao nhất là 20.000.000 VNĐ/tháng. Lương thử việc = 85% x 20.000.000 = 17.000.000 VNĐ/tháng.

**Got:** Lương thử việc của nhân viên Junior mức cao nhất là 17.000.000 VNĐ/tháng.

### Error Tree walkthrough

1. **Output đúng?** → So sánh `Got` với `Expected` và metric thấp nhất.
2. **Context đúng?** → Worst metric hiện tại là `faithfulness`; dùng metric này để xác định retrieval hay generation là nút lỗi chính.
3. **Root cause:** Generation failure: answer contains claims insufficiently supported by retrieved context.
4. **Fix ở bước:** Tighten the grounded-answer prompt, reduce generation freedom, and ensure supporting context is included.

### Nếu có thêm 1 giờ

- Chạy lại full RAGAS sau mỗi thay đổi và chỉ chấp nhận submission khi quality gate đạt.
- Thêm regression test cho grounded concise answer format và numeric-policy prompt.
- Nếu lỗi thuộc version conflict, ưu tiên metadata/version filtering thay vì hard-code tên tài liệu.
