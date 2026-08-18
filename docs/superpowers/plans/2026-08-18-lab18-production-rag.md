# Lab 18 Production RAG Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Hoàn thiện 5 module Production RAG, pipeline/evaluation artifacts, reflection và GitHub Actions để tối đa rubric 100 + 10 bonus.

**Architecture:** Giữ API scaffold, bổ sung implementation production-grade theo chuỗi hierarchical chunking -> combined enrichment -> BM25+dense+RRF -> CrossEncoder -> grounded answer -> RAGAS. GitHub Actions là môi trường verification chính vì runtime hiện tại không truy cập mạng để clone/tải model.

**Tech Stack:** Python 3.11, sentence-transformers, BGE-M3, Qdrant, underthesea, rank-bm25, OpenAI, RAGAS, pytest, ruff, GitHub Actions.

**Spec:** `docs/superpowers/specs/2026-08-18-lab18-production-rag-design.md`

## Global Constraints

- Không hard-code `test_set.json` answers vào pipeline.
- Secrets chỉ qua env (`OPENAI_API_KEY`, `HF_TOKEN`).
- Public function/class signatures phải tương thích tests hiện có.
- Unit tests phải chạy khi không có OpenAI key.
- Full evaluation dùng Qdrant service container.

---

### Task 1: Establish CI RED baseline

**Files:**
- Create: `.github/workflows/ci.yml`
- Test: `tests/test_m1.py` ... `tests/test_m5.py`

**Interfaces:**
- Consumes: repository starter scaffold.
- Produces: PR-triggered pytest/ruff/TODO/compile verification.

- [x] **Step 1:** Tạo workflow Python 3.11 có pip/HF cache.
- [x] **Step 2:** Chạy public tests trên PR để xác nhận starter đang RED.
- [ ] **Step 3:** Đọc failed job logs và ghi baseline failure evidence.

### Task 2: Implement M1 chunking

**Files:**
- Modify: `src/m1_chunking.py`
- Test: `tests/test_m1.py`

**Interfaces:**
- Produces: `chunk_semantic(...) -> list[Chunk]`, `chunk_hierarchical(...) -> tuple[list[Chunk], list[Chunk]]`, `chunk_structure_aware(...) -> list[Chunk]`.

- [ ] **Step 1:** Dùng public tests hiện có làm RED assertions cho non-empty semantic, valid parent IDs, smaller children, section metadata.
- [ ] **Step 2:** Implement sentence splitting + cosine grouping với lazy sentence-transformer.
- [ ] **Step 3:** Implement paragraph-safe parent/child chunker, split oversized units.
- [ ] **Step 4:** Implement Markdown header parser preserving header text.
- [ ] **Step 5:** Run `pytest tests/test_m1.py -v` trên Actions và sửa đến GREEN.

### Task 3: Implement M2 hybrid search

**Files:**
- Modify: `src/m2_search.py`
- Test: `tests/test_m2.py`

**Interfaces:**
- Produces: Vietnamese segmenter, `BM25Search`, `DenseSearch`, `reciprocal_rank_fusion`.

- [ ] **Step 1:** Dùng BM25/RRF tests hiện có làm RED.
- [ ] **Step 2:** Implement underthesea segmentation và BM25 index/search.
- [ ] **Step 3:** Implement Qdrant collection/index/query_points dense retrieval.
- [ ] **Step 4:** Implement deterministic RRF dedupe/sort.
- [ ] **Step 5:** Run `pytest tests/test_m2.py -v` trên Actions và sửa đến GREEN.

### Task 4: Implement M3 reranking

**Files:**
- Modify: `src/m3_rerank.py`
- Test: `tests/test_m3.py`

**Interfaces:**
- Produces: lazy `CrossEncoderReranker._load_model()`, ranked `RerankResult` list.

- [ ] **Step 1:** Dùng relevance/sort/top-k tests hiện có làm RED.
- [ ] **Step 2:** Load `sentence_transformers.CrossEncoder` lazily.
- [ ] **Step 3:** Predict pairs, normalize scalar/array scores, stable sort descending.
- [ ] **Step 4:** Run `pytest tests/test_m3.py -v` và benchmark assertions trên Actions.

### Task 5: Implement M4 RAGAS + failure analysis

**Files:**
- Modify: `src/m4_eval.py`
- Test: `tests/test_m4.py`

**Interfaces:**
- Produces: numeric aggregate metrics, `per_question: list[EvalResult]`, bottom-N diagnostic dictionaries.

- [ ] **Step 1:** Dùng metric-key/diagnosis tests hiện có làm RED.
- [ ] **Step 2:** Implement RAGAS Dataset/evaluate conversion và robust fallback.
- [ ] **Step 3:** Implement Diagnostic Error Tree and bottom-N ordering.
- [ ] **Step 4:** Make report writer directory-safe and JSON-serializable.
- [ ] **Step 5:** Run `pytest tests/test_m4.py -v` trên Actions.

### Task 6: Implement M5 enrichment + combined bonus

**Files:**
- Modify: `src/m5_enrichment.py`
- Test: `tests/test_m5.py`

**Interfaces:**
- Produces: four individual techniques and `_enrich_single_call(text, source) -> dict`.

- [ ] **Step 1:** Dùng public enrichment tests làm RED behavior.
- [ ] **Step 2:** Implement deterministic no-key fallbacks.
- [ ] **Step 3:** Implement OpenAI individual techniques.
- [ ] **Step 4:** Implement single-call JSON enrichment with code-fence-safe parsing.
- [ ] **Step 5:** Ensure combined fallback enriches with source/context.
- [ ] **Step 6:** Run `pytest tests/test_m5.py -v` trên Actions.

### Task 7: Production pipeline, parent expansion and latency bonus

**Files:**
- Modify: `src/pipeline.py`
- Modify: `naive_baseline.py`
- Modify: `main.py`
- Modify: `check_lab.py`
- Create: `src/reporting.py`

**Interfaces:**
- Produces: end-to-end pipeline, parent-context expansion, grounded generation, timing JSON/Markdown-friendly data and consistent `reports/` paths.

- [ ] **Step 1:** Add timing collector/report helper.
- [ ] **Step 2:** Preserve parent map during indexing and expand reranked children to parents for answer context.
- [ ] **Step 3:** Strengthen grounded/version-aware answer prompt without ground-truth leakage.
- [ ] **Step 4:** Save baseline/production/latency reports directly into `reports/`.
- [ ] **Step 5:** Align `check_lab.py` with assignment-required M5/reflection/report artifacts and make validation exit non-zero on real errors.
- [ ] **Step 6:** Run compile + tests + check_lab smoke via Actions.

### Task 8: Analysis deliverables

**Files:**
- Modify/Create: `analysis/failure_analysis.md`
- Create: `analysis/reflections/reflection_ChuNguyenTuanAnh.md`

**Interfaces:**
- Consumes: real CI/evaluation evidence and project context.
- Produces: full rubric analysis/reflection.

- [ ] **Step 1:** Generate bottom-5 table structure from report failures; populate real metrics when full evaluation runs.
- [ ] **Step 2:** Map all 5 lecture concepts to concrete functions and observations.
- [ ] **Step 3:** Record exact CI errors encountered and fixes actually applied.
- [ ] **Step 4:** Add ContractAI action plan: current React/FastAPI/OpenAI RAG flow, hybrid retrieval/reranking/RAGAS/enrichment plan and timeline.

### Task 9: Full GitHub evaluation workflow

**Files:**
- Create: `.github/workflows/full-evaluation.yml`
- Create: `scripts/check_rubric.py`

**Interfaces:**
- Produces: manual full baseline+production RAGAS run and score/bonus summary artifacts.

- [ ] **Step 1:** Configure Qdrant service and Python/HF caches.
- [ ] **Step 2:** Pass `OPENAI_API_KEY` and `HF_TOKEN` only through secrets.
- [ ] **Step 3:** Run `python main.py`, `python check_lab.py`, schema/score checker.
- [ ] **Step 4:** Upload `reports/` + `analysis/` regardless of quality-gate outcome.
- [ ] **Step 5:** Verify run logs/artifacts and fix integration failures.

### Task 10: Final verification and PR readiness

**Files:** all changed files.

- [ ] **Step 1:** Confirm public tests 100% pass.
- [ ] **Step 2:** Confirm ruff and zero-TODO gates pass.
- [ ] **Step 3:** Confirm full pipeline exits 0 with Qdrant/OpenAI secrets.
- [ ] **Step 4:** Review RAGAS metrics honestly against bonus thresholds.
- [ ] **Step 5:** Review PR diff for secret leakage and ground-truth hard-coding.
- [ ] **Step 6:** Mark draft PR ready only after verification evidence is available.