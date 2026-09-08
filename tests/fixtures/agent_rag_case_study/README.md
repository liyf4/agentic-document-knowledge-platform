# Agent/RAG long-document case study

This folder is the human-auditable source of truth for the per-case RAG benchmark.

- `documents/` contains the documents uploaded to one isolated test session.
- `cases.jsonl` contains every question, standard answer, required evidence ID, and deterministic answer checks.
- Evidence IDs such as `[EVIDENCE:NS-042]` are stable labels for evaluation. Questions never contain these IDs.

Metrics:

- **CrossEncoder Accuracy@1**: `1` when the final Rank-1 after CrossEncoder reranking contains a required evidence ID.
- **RRF Accuracy@4**: `1` when the pre-rerank RRF Top-4 covers every required evidence ID; evidence coverage is also recorded.
- **Answer Accuracy**: fraction of standard-answer fact groups found in the real Agent/ZAI answer. `answer_passed` additionally requires structured citations.

The suite also records evidence-source accuracy, section-path and chunk-ID completeness, deterministic metadata fact recall, fact-to-evidence linkage, citation presence, and citation-to-gold-evidence coverage. Routing metrics are intentionally excluded.

Generated answers and comparisons are written to `test-results/agent-rag/case-study/` and are intentionally not version controlled.
