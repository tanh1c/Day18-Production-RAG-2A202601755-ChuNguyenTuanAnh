# Group Report — Lab 18: Production RAG

**Nhóm:** Cá nhân — Chu Nguyễn Tuấn Anh  
**Ngày:** 18/08/2026

## Thành viên & Phân công

| Tên | Module | Hoàn thành | Tests pass |
|---|---|:---:|:---:|
| Chu Nguyễn Tuấn Anh | M1: Advanced Chunking | ✅ | ✅ |
| Chu Nguyễn Tuấn Anh | M2: Hybrid Search | ✅ | ✅ |
| Chu Nguyễn Tuấn Anh | M3: Cross-encoder Reranking | ✅ | ✅ |
| Chu Nguyễn Tuấn Anh | M4: RAGAS Evaluation & Failure Analysis | ✅ | ✅ |
| Chu Nguyễn Tuấn Anh | M5: Contextual Enrichment / HyQA | ✅ | ✅ |

## Kết quả RAGAS

| Metric | Naive | Production | Δ |
|---|---:|---:|---:|
| Faithfulness | 0.8183 | **0.8958** | **+0.0775** |
| Answer Relevancy | 0.7662 | **0.8401** | **+0.0739** |
| Context Precision | 0.9250 | **0.9500** | **+0.0250** |
| Context Recall | 0.9250 | **0.9750** | **+0.0500** |

**Rubric evidence cuối:** Base **100/100** + Bonus **10/10** = **110/110**.

## Key Findings

1. **Biggest improvement:** Faithfulness tăng mạnh nhất, từ `0.8183` lên `0.8958` (+0.0775). Hai thay đổi đóng góp rõ nhất là giảm generation thừa bằng grounded concise-answer policy và dùng deterministic arithmetic guardrail cho policy có phép tính phần trăm theo thời gian.
2. **Biggest challenge:** Corpus có policy cũ/mới cùng tồn tại và một số câu hỏi nhiều ý. Nếu retrieve một query duy nhất, pipeline có thể lấy đúng tài liệu cho ý thứ nhất nhưng bỏ mất tài liệu của ý thứ hai. Facet-aware query decomposition + hybrid retrieval + cross-encoder reranking giúp khắc phục lỗi multi-hop này.
3. **Surprise finding:** Retrieval tốt chưa đảm bảo faithfulness cao. Có case context precision/recall đạt 1.0 nhưng generation vẫn tính sai hoặc thêm claim không cần thiết. Việc tách retrieval failure khỏi generation failure bằng RAGAS/Error Tree giúp sửa đúng tầng thay vì chỉ tăng top-k.

## Failure Case Study

**Question:** Nhân viên tạm ứng 15 triệu, sau 20 ngày mới thanh toán. Bị phạt bao nhiêu?

- Policy context: hoàn ứng trong 15 ngày; quá hạn tính phí 2%/tháng trên số tiền chưa hoàn ứng.
- Sai sót ban đầu: LLM áp thẳng 2% lên 15 triệu và trả `300.000 VNĐ`, bỏ qua việc chỉ quá hạn 5 ngày.
- Fix: parse amount, elapsed days, grace period và monthly rate từ query/context; tính pro-rata bằng code: `15.000.000 × 2% × 5/30 = 50.000 VNĐ`.
- Kết quả: đáp án cuối trả đúng khoảng `50.000 VNĐ`; aggregate faithfulness vượt ngưỡng bonus 0.85.

## Presentation Notes (5 phút)

1. **RAGAS scores:** Production đạt Faithfulness `0.8958`, Answer Relevancy `0.8401`, Context Precision `0.9500`, Context Recall `0.9750`; cả bốn đều > 0.75.
2. **Biggest win:** M2 + M3 + parent expansion giữ recall/precision cao, còn M5 enrichment giữ version/effective-date metadata để xử lý policy conflict.
3. **Case study:** dùng câu tạm ứng 15 triệu để minh họa Error Tree: context đúng nhưng answer sai → lỗi generation/arithmetic → deterministic guardrail.
4. **Next optimization nếu có thêm 1 giờ:** thêm temporal/version filter trước rerank để giảm context cũ ở các câu policy có nhiều revision; benchmark recall@k và faithfulness variance qua nhiều lần eval.
