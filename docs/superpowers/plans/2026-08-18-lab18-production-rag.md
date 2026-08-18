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
- Workflows chỉ chạy thủ công (`workflow_dispatch`) để tránh spam Actions.

---

## Trạng thái triển khai

- [x] M1 Semantic / Hierarchical / Structure-aware chunking.
- [x] M2 BM25 Vietnamese + BGE-M3/Qdrant + RRF.
- [x] M3 CrossEncoder reranking + benchmark.
- [x] M4 RAGAS + Diagnostic Error Tree.
- [x] M5 combined enrichment 1 call/chunk + fallback.
- [x] Parent-context expansion + latency report.
- [x] Reflection cá nhân + CI evidence.
- [x] Fast CI: 37/37 starter tests pass, Ruff/compile/TODO gates pass.
- [x] Full evaluation baseline: core 100/100, bonus 7/10; faithfulness 0.8283 là remaining quality gap.
- [ ] Final grounded-generation fix verified with faithfulness >= 0.85.
- [ ] Final reports/failure analysis persisted back into submission branch.
- [ ] Final rubric summary confirms 110/110.

## Final verification procedure

1. Chạy **Full RAG Evaluation** thủ công đúng một lần trên branch `feat/lab18-submission`.
2. Workflow chạy tests + Ruff + Qdrant + baseline + production + RAGAS.
3. `scripts/check_rubric.py --strict` chỉ exit 0 khi base = 100/100 và bonus = 10/10.
4. Dù quality gate pass/fail, workflow persist report/failure-analysis thực tế về branch submission để debug có bằng chứng.
5. Chỉ coi bài hoàn tất khi workflow success và `reports/rubric_summary.md` ghi `110/110`.