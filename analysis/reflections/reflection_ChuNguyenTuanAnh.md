# Reflection Lab 18 — Chu Nguyễn Tuấn Anh

## Phần 1 — Mapping lecture concept → code

| Lecture Concept | Module | Hàm cụ thể | Observation |
|---|---|---|---|
| Semantic chunking | M1 | `chunk_semantic()` | Thay vì cắt theo số ký tự cố định, pipeline encode các câu liên tiếp và dùng cosine similarity để quyết định boundary. Threshold cao tạo boundary chặt hơn; fallback lexical chỉ dùng khi model không tải được. |
| Hierarchical chunking | M1 | `chunk_hierarchical()` | Child nhỏ được dùng để retrieval chính xác, nhưng pipeline giữ `parent_key` để mở rộng lại parent context trước generation. Cách này giảm tình trạng một fact bị cắt khỏi điều kiện/ngoại lệ ở đoạn kế bên. |
| BM25 + Dense fusion | M2 | `BM25Search.search()`, `DenseSearch.search()`, `reciprocal_rank_fusion()` | BM25 bắt tốt keyword/số/chính sách; BGE-M3 bắt semantic paraphrase. RRF hợp nhất rank mà không cần chuẩn hóa raw BM25 score và cosine score về cùng thang. |
| Cross-encoder reranking | M3 | `CrossEncoderReranker.rerank()` | Bi-encoder/dense dùng cho candidate recall; cross-encoder đọc trực tiếp cặp query-document để tăng precision ở top-3. Latency cao hơn được đo riêng trong `reports/latency_report.md`. |
| RAGAS 4 metrics | M4 | `evaluate_ragas()` | Faithfulness đo answer có bám context; answer relevancy đo mức trả lời đúng câu hỏi; context precision/recall tách lỗi retrieval nhiễu và retrieval bỏ sót. Bottom-5 được map vào Diagnostic Error Tree thay vì chỉ liệt kê score. |
| Contextual enrichment / HyQA | M5 | `_enrich_single_call()` | Một call/chunk lấy summary, hypothesis questions, context và metadata. Enrichment giữ số liệu, phủ định, ngày hiệu lực và thông tin superseded để dense retrieval phân biệt policy cũ/mới tốt hơn, đồng thời tiết kiệm API call so với 4 call riêng. |

### Điểm mình rút ra từ corpus

Corpus lab cố tình có tài liệu cũ và mới cùng tồn tại, ví dụ nghỉ phép năm 2023/2024 và password policy v1/v2. Vì vậy retrieval relevance đơn thuần chưa đủ: metadata nguồn/phiên bản và parent context phải được giữ tới bước answer. Prompt generation chỉ ưu tiên phiên bản hiện hành khi chính context có bằng chứng như ngày hiệu lực mới hơn hoặc câu “thay thế phiên bản cũ”; không dùng `test_set.json` để hard-code đáp án.

## Phần 2 — Khó khăn & cách giải quyết

### 1. Starter scaffold chưa implement nên public tests fail ngay

Ví dụ M1 ban đầu trả `[]` cho semantic chunking, trong khi test có assertion:

```text
AssertionError: Semantic chunking should return chunks
```

Cách debug:

1. Đọc test trước để xác định invariant thực sự: non-empty, `Chunk` type, parent ID hợp lệ, child nhỏ hơn parent, header/section metadata.
2. Implement theo invariant thay vì chỉ sửa để một input cụ thể pass.
3. Thêm xử lý text rỗng, paragraph quá dài và fallback model để tránh hidden edge case.
4. Đưa pytest vào GitHub Actions để mỗi commit được kiểm tra lại.

### 2. Môi trường thực thi hiện tại không clone GitHub trực tiếp được

Exact error khi thử tạo local workspace:

```text
fatal: unable to access 'https://github.com/tanh1c/Day18-Production-RAG-2A202601755-ChuNguyenTuanAnh.git/': Could not resolve host: github.com
```

Cách debug/giải quyết:

- Xác định đây là DNS/network restriction của runtime, không phải lỗi repo.
- Chuyển sang GitHub-native workflow: tạo branch/PR bằng GitHub connector, push file qua Contents API và dùng GitHub Actions làm test runner thật.
- Thêm `concurrency.cancel-in-progress` để tránh mỗi commit tạo một runner backlog riêng.

### 3. Version conflict làm RAG khó hơn demo retrieval thông thường

Nếu query “bao lâu đổi mật khẩu” retrieve cả v1 (90 ngày) và v2 (120 ngày), cosine relevance có thể xem cả hai đều rất liên quan. Nếu chỉ lấy top-1, answer có thể đúng ngữ nghĩa nhưng sai policy hiện hành.

Giải pháp:

- Preserve `source`, `parent_id`, `parent_key` xuyên suốt enrichment/index/retrieval.
- Hybrid retrieve top candidates, cross-encoder rerank, sau đó mở rộng child → parent.
- Context enrichment nhấn mạnh version/effective/superseded information.
- Grounded prompt ưu tiên văn bản hiện hành dựa trên evidence trong context, không dựa trên tên câu hỏi trong test set.

### Kiến thức mình cần củng cố

- Cách tune chunk size/semantic threshold theo corpus thay vì dùng một default cho mọi project.
- RAGAS metric variance và cách thiết kế evaluation set đủ đại diện.
- Retrieval versioning/temporal metadata khi knowledge base có nhiều revision.
- Cost/latency trade-off giữa enrichment offline và query-time reranking.

## Phần 3 — Action Plan áp dụng vào project cá nhân

## Project: ContractAI

### Hiện tại

ContractAI là web hỗ trợ rà soát hợp đồng bằng AI. Flow hiện tại gồm React/Vite frontend → FastAPI backend → upload PDF/DOCX → extract text → gửi context sang OpenAI → structured JSON cho clauses/review items/risks → map kết quả về page/paragraph → Q&A theo context → export report.

Known issues khi phát triển RAG/Q&A sâu hơn:

- Contract dài có thể mất context nếu chỉ chia chunk cố định.
- Legal terms có cả exact wording và paraphrase, nên dense-only hoặc keyword-only đều có blind spot.
- Risk analysis cần evidence rất chính xác; context nhiễu dễ làm LLM suy diễn.
- Chưa có evaluation harness định lượng retrieval/generation regression khi đổi prompt/model/chunking.

### Plan áp dụng

1. [ ] **Chunking strategy:** structure-aware + hierarchical. Parse heading/điều/khoản trước, child khoảng 300–500 tokens để retrieve, parent là toàn điều khoản/section để đưa vào analysis.
2. [ ] **Search:** hybrid BM25 + dense. BM25 ưu tiên số điều, tên bên, thuật ngữ pháp lý; dense xử lý câu hỏi paraphrase; fuse bằng RRF.
3. [ ] **Reranking:** có. Dùng multilingual cross-encoder trên top 15–20 candidates, trả top 3–5 evidence chunks cho risk/Q&A.
4. [ ] **Evaluation:** xây ContractAI eval set theo nhóm clause lookup, negation, conflict, numeric/date, cross-clause và missing-information. Theo dõi faithfulness, context precision/recall, answer relevancy + custom citation correctness.
5. [ ] **Enrichment:** contextual prepend + metadata extraction offline khi ingest contract. Metadata gồm section, clause type, party, effective date, monetary amount, obligation/prohibition và page/paragraph locator.
6. [ ] **Evidence policy:** mọi risk/answer phải mang `source locator` về paragraph/page gốc; nếu evidence không đủ thì model phải abstain thay vì tự suy diễn.
7. [ ] **Observability:** log latency theo ingestion/retrieval/rerank/generation và lưu eval report theo commit để detect regression trong CI.

### Timeline

- **Tuần 1:** refactor ingestion sang structure-aware + hierarchical chunking; tạo metadata/page locator chuẩn.
- **Tuần 2:** thêm BM25 + dense hybrid retrieval và benchmark recall@k trên ContractAI eval set.
- **Tuần 3:** thêm cross-encoder reranking, evidence expansion và citation correctness tests.
- **Tuần 4:** tích hợp RAGAS/custom evaluation vào CI; đặt quality gate cho faithfulness/context recall.
- **Tuần 5:** thử combined enrichment offline, đo delta quality/cost/latency và chỉ giữ technique có improvement rõ ràng.

## Kết luận

Điểm quan trọng nhất của Lab 18 với mình là chuyển từ “LLM trả lời được” sang một pipeline có thể đo và debug theo từng tầng: chunking → retrieval → reranking → generation → evaluation. Với ContractAI, cách tiếp cận này đặc biệt quan trọng vì một answer nghe hợp lý nhưng lấy sai điều khoản hoặc sai phiên bản vẫn là failure.