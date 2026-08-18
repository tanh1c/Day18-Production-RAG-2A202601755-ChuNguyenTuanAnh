# Lab 18 Production RAG — Design

## Goal

Hoàn thiện toàn bộ Lab 18 theo rubric 100 điểm và triển khai đầy đủ các bonus khả thi: combined enrichment, RAGAS quality gates và latency breakdown. Toàn bộ verification phải chạy được trên GitHub Actions.

## Constraints

- Python 3.11.
- Không hard-code câu trả lời từ `test_set.json` vào retrieval/generation.
- `OPENAI_API_KEY` và `HF_TOKEN` chỉ được đọc qua environment/GitHub Secrets.
- Qdrant chạy bằng service container trong GitHub Actions.
- Pipeline phải có fallback hợp lý khi không có OpenAI key để unit tests vẫn chạy.
- Các model theo scaffold: BGE-M3 cho dense retrieval và `BAAI/bge-reranker-v2-m3` cho reranking.
- Báo cáo chuẩn nằm dưới `reports/`; analysis nằm dưới `analysis/`.

## Architecture

Luồng production:

`documents -> hierarchical chunking -> combined enrichment -> BM25 + dense retrieval -> RRF -> cross-encoder reranking -> grounded answer -> RAGAS -> failure analysis/report`

Các module giữ API sẵn có để tương thích public grader.

## M1 — Chunking

- `chunk_semantic`: tách câu, encode bằng sentence-transformer, nhóm các câu liên tiếp khi cosine similarity đạt threshold.
- `chunk_hierarchical`: tạo parent chunks theo paragraph, sau đó child chunks nhỏ hơn; mỗi child giữ `parent_id` hợp lệ.
- `chunk_structure_aware`: parse Markdown headers, giữ header trong text và lưu `section` metadata.
- Các helper phải xử lý text rỗng và paragraph dài mà không tạo chunk rỗng.

## M2 — Search

- Vietnamese segmentation qua `underthesea.word_tokenize(..., format="text")`, sau đó thay `_` thành khoảng trắng.
- BM25 dùng `BM25Okapi`, bỏ kết quả score <= 0.
- Dense dùng BGE-M3 + Qdrant `query_points()`.
- RRF deduplicate theo text và cộng `1 / (k + rank + 1)`; output method=`hybrid`.
- Metadata nguồn tài liệu phải được bảo toàn.

## M3 — Reranking

- Lazy-load `sentence_transformers.CrossEncoder`.
- Predict theo cặp `(query, document)`; sort giảm dần; trả `RerankResult` tối đa `top_k`.
- Benchmark giữ API có sẵn và được dùng trong latency report.

## M4 — Evaluation

- `evaluate_ragas` chạy 4 metrics trong try/except và luôn trả 4 numeric aggregate keys.
- Khi RAGAS chạy được, tạo `EvalResult` cho từng câu hỏi để phục vụ failure analysis.
- Khi RAGAS lỗi/không có key, trả zero aggregates + empty per-question thay vì crash unit tests.
- `failure_analysis` tính average score, xác định metric thấp nhất và map qua Diagnostic Error Tree thành diagnosis/fix.
- `save_report` đảm bảo parent directory tồn tại.

## M5 — Enrichment

- Implement 4 technique riêng có deterministic fallback.
- Combined mode `_enrich_single_call` thực hiện đúng một OpenAI request/chunk và trả `summary`, `questions`, `context`, `metadata`.
- Combined fallback vẫn tạo enrichment khác raw text khi có source/context để pipeline chạy hữu ích offline.
- JSON response parsing phải chịu được markdown code fence.

## Pipeline quality

- Hierarchical parent text được lưu theo `parent_id`; retrieval trên child nhưng answer context có thể mở rộng về parent để tăng recall.
- Prompt answer yêu cầu chỉ dùng context, ưu tiên văn bản hiện hành khi có policy cũ/mới và nêu rõ xung đột nếu cần.
- Không dùng ground truth trong generation.
- Timing collector ghi load/chunk, enrichment, indexing, retrieval, reranking, generation, RAGAS và total.

## Reports and analysis

- `reports/ragas_report.json`
- `reports/naive_baseline_report.json`
- `reports/latency_report.json`
- `analysis/failure_analysis.md` sinh từ bottom-5 thực tế khi có RAGAS data.
- `analysis/reflections/reflection_ChuNguyenTuanAnh.md` map đủ 5 module, ghi debug evidence từ CI và action plan áp dụng vào ContractAI.

## CI

### Fast CI

Mỗi push/PR:
- Python 3.11
- dependency + Hugging Face cache
- pytest
- ruff
- zero `# TODO:` gate
- compileall

### Full evaluation

Manual workflow (`workflow_dispatch`) và có thể chạy trên branch:
- Qdrant service container
- `OPENAI_API_KEY`, optional `HF_TOKEN`
- chạy baseline + production
- `check_lab.py`
- validate report schema
- tính rubric/bonus gates
- upload `reports/` và `analysis/` artifacts

RAGAS quality targets cho bonus:
- faithfulness >= 0.85
- answer_relevancy >= 0.75
- context_precision >= 0.75
- context_recall >= 0.75

Các threshold là mục tiêu chất lượng; workflow phải báo rõ metric đạt/chưa đạt thay vì giả mạo score.