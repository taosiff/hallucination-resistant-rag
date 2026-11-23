# EvidenceGuard RAG: Comprehensive Project Plan & Implementation Roadmap

**Project Title:** EvidenceGuard: Hallucination-Resistant Retrieval-Augmented Generation Architecture  
**Author:** EvidenceGuard Engineering & Research Team  
**Version:** 1.0.0-PROD  
**Target Completion:** End-to-End Functional, Evaluated, and Production-Ready RAG System  

---

## 1. Executive Summary & Core Objective

Standard RAG architectures optimize almost exclusively for generation fluency and cosine similarity. Under real-world conditions, however, retrieval systems frequently return:
1. Irrelevant or distracting snippets.
2. Documents that do not contain the answer.
3. Partial or fragmented evidence.
4. Mutually contradictory claims across sources or dates.
5. Injected or misleading adversarial instructions.

EvidenceGuard is engineered to solve the **epistemic grounding problem** in RAG: teaching the system when to answer, when to qualify an answer with explicit caveats, when to highlight contradictory sources, and when to abstain entirely.

---

## 2. Phase-by-Phase Execution Roadmap

```
Phase 0: Environment & Hardware Setup [COMPLETED]
   │
Phase 1: Deep Research & Literature Review [COMPLETED]
   │
Phase 2: Benchmark Dataset Construction & Curation (Categories A-H)
   │
Phase 3: Baseline RAG Implementation & Ingestion Engine
   │
Phase 4: LangGraph Agentic Workflow & State Machine
   │
Phase 5: Evidence Sufficiency & Calibrated Abstention Classifiers
   │
Phase 6: Advanced Retrieval & Reranking Comparative Experiments
   │
Phase 7: Fine-Tuning Experiments (LoRA/QLoRA & Noisy Retrieval Robustness)
   │
Phase 8: Conflict Resolution & Claim-Level Citation Verification
   │
Phase 9: Comprehensive Benchmark Evaluation & Ablation Suite
   │
Phase 10: Experiment Tracking (MLflow) & Graph Tracing (LangSmith)
   │
Phase 11: Production REST API (FastAPI) & Interactive Frontend (React)
   │
Phase 12: Rigorous Test Suite, Security Hardening & Edge Cases
   │
Phase 13: Repository Hygiene, Dockerization & CI Readiness
   │
Phase 14: Comprehensive Technical Documentation Suite
   │
Phase 15: Research Synthesis & Empirical Contributions Validation
   │
Phase 16: Final Review, Verification & Delivery
```

---

## 3. Detailed Phase Breakdown & Milestones

### Phase 0: Hardware Profiling & Compute Strategy [Status: Done]
* **Deliverables:** `HARDWARE_REPORT.md`, `TECH_STACK.md`, `PROJECT_PLAN.md`.
* **Hardware Verified:** 20-Core i7-14700K, 64 GB RAM, RTX 4080 SUPER (16 GB VRAM), ~650 GB SSD free space.
* **Execution Boundary:** Full local retrieval/reranking/classification + hybrid local/API LLM options.

### Phase 1: Literature Review & Architecture Decisions [Status: Done]
* **Deliverables:**
  - `research/LITERATURE_REVIEW.md`
  - `research/PAPER_COMPARISON.md`
  - `research/ARCHITECTURE_DECISION.md`
  - `research/IMPLEMENTATION_REFERENCES.md`
* **Deep Review Of:** Self-RAG (ICLR 2024), CRAG (2024), RAFT (2024), RAGTruth (ACL 2024), RGB Benchmark (AAAI 2024), ARES (NAACL 2024), SQuAD 2.0 unanswerability, and conformal risk control for abstention.

### Phase 2: Dataset & Evaluation Benchmark Construction
* **Target Categories:**
  - **Category A:** Answerable queries with complete gold evidence.
  - **Category B:** Unanswerable queries (answer missing from corpus).
  - **Category C:** Irrelevant retrieval (semantic distractor chunks).
  - **Category D:** Partial evidence (some sub-questions supported, some unsupported).
  - **Category E:** Conflicting evidence (inconsistent facts across passages).
  - **Category F:** Multi-hop questions (evidence distributed across multiple passages).
  - **Category G:** Outdated evidence (temporal contradiction with newer source).
  - **Category H:** Misleading / adversarial context (counterfactual noise).
* **Deliverables:** `data/benchmark_schema.json`, `data/splits/{train,val,test}.json`, `data/DATASET_CARD.md`, `data/README.md`.
* **Discipline:** No fabricated metrics, no leakage between train and test document splits.

### Phase 3: Baseline RAG Implementation
* **Document Ingestion:** Multi-format parser (PDF, DOCX, TXT, MD, CSV) with text normalization and source tracking.
* **Chunking Strategies:** Fixed-size, recursive character splitting, semantic chunking, and parent-child chunking.
* **Indexing:** Dense embeddings (`bge-large-en-v1.5` / `all-MiniLM-L6-v2`) via ChromaDB / FAISS + Sparse BM25 index.
* **Deliverables:** `backend/ingestion/`, `backend/retrieval/baseline_retriever.py`, baseline query pipeline.

### Phase 4: LangGraph Agentic Architecture
* **State Object:** Strictly typed Pydantic state (`GraphState`) holding original query, reformulated queries, retrieved documents, reranker scores, sufficiency assessment, conflict reports, generated answer, extracted claims, citation checks, and execution trace.
* **Nodes:**
  1. `analyze_query`
  2. `rewrite_query`
  3. `retrieve_documents`
  4. `rerank_documents`
  5. `assess_evidence`
  6. `detect_conflicts`
  7. `decide_answerability`
  8. `generate_answer`
  9. `verify_claims`
  10. `validate_citations`
  11. `abstain`
  12. `finalize_response`
* **Deliverables:** `backend/graph/state.py`, `backend/graph/nodes.py`, `backend/graph/workflow.py`.

### Phase 5: Evidence Sufficiency & Calibrated Abstention Model
* **Task:** Classify retrieval state into `[SUFFICIENT, PARTIALLY_SUFFICIENT, INSUFFICIENT, CONFLICTING, IRRELEVANT]`.
* **Feature Pipeline:** 12 statistical, distributional, lexical, and semantic features extracted from top-$k$ retrieved chunks.
* **Models:** Logistic Regression, Random Forest, LightGBM, XGBoost, and DeBERTa cross-encoder.
* **Evaluation:** Accuracy, Macro-F1, AUROC, Expected Calibration Error (ECE), Brier Score, and Risk-Coverage curves.
* **Deliverables:** `backend/evidence/classifier.py`, `backend/evidence/features.py`, trained models in `models/`.

### Phase 6: Advanced Retrieval Experiments
* Comparative runs:
  - Exp A: Query Rewriting (Original vs LLM Reformulation vs Multi-Query vs HyDE).
  - Exp B: Cross-Encoder Reranking (None vs `ms-marco-MiniLM-L-6-v2` vs `bge-reranker-base`).
  - Exp C: Hybrid Retrieval (Dense only vs BM25 only vs Hybrid RRF vs Hybrid RRF + Reranker).
  - Exp D: Chunking variations (256, 512, 1024 tokens + overlap vs Semantic).
* **Deliverables:** `experiments/retrieval/`, empirical comparison logs and tables.

### Phase 7: Fine-Tuning Experiments
* **Exp 1:** Prompt-only baseline with evidence grounding instructions.
* **Exp 2:** Fine-tuned answerability classifier.
* **Exp 3 & 4:** QLoRA / LoRA fine-tuning on noisy retrieval scenarios (incorporating RAFT and Self-RAG principles: learning to cite verbatim, ignore distractors, and explicitly decline when evidence is lacking).
* **Deliverables:** `experiments/finetuning/`, fine-tuning scripts, training loss curves, evaluation comparisons.

### Phase 8: Conflicting Evidence & Citation Verification
* **Contradiction Detection:** Pairwise NLI across retrieved top passages to flag factual disagreements.
* **Claim-Level Deconstruction:** Segment generated answers into atomic assertions; check each claim against cited passages via entailment scoring.
* **Citation Validator:** Strict validation ensuring citations point to valid chunk IDs, accurate offsets, and true entailment.
* **Deliverables:** `backend/verification/claim_verifier.py`, `backend/verification/conflict_detector.py`.

### Phase 9: Evaluation Benchmark & Ablation Studies
* Quantitative measurement across all 7 systems:
  - Retrieval: Recall@k, Precision@k, MRR, nDCG@k.
  - Answer Quality: Exact Match, F1, Semantic Groundedness.
  - Faithfulness: Claim support rate, Context precision/recall, Citation accuracy.
  - Abstention: Abstention F1, False Answer Rate on unanswerable queries, Selective Risk, Risk-Coverage AUC.
* Full ablation study removing individual components.
* **Deliverables:** `backend/evaluation/`, `experiments/ablations/`, final comparison table.

### Phase 10: Experiment Tracking & Observability
* MLflow integration for model artifacts, parameters, and evaluation runs.
* LangSmith & OpenTelemetry tracing hooks for graph nodes and token latency.
* **Deliverables:** `backend/core/telemetry.py`, MLflow experiment logs.

### Phase 11: Production Application (FastAPI + React)
* FastAPI REST API with endpoints (`/health`, `/documents/upload`, `/query`, `/query/evaluate`, `/retrieve`, `/experiments`, `/models`).
* React/Vite modern responsive dashboard with live inspection drawer, metric charts, and document management.
* **Deliverables:** `backend/api/`, `frontend/`.

### Phase 12: Testing, Security & Hardening
* Unit, integration, and end-to-end tests covering all edge cases (empty docs, corrupted PDFs, zero retrieved passages, prompt injection in chunks).
* **Deliverables:** `tests/test_*.py` suite with passing pytest execution.

### Phase 13: Repository Structure & Dockerization
* Clean production layout, `.env.example`, `requirements.txt`, `Dockerfile`, `docker-compose.yml`.

### Phase 14: Comprehensive Documentation
* 10 core documentation files covering Architecture, Research, Benchmark, Model Card, API, Installation, and Limitations.

### Phase 15 & 16: Research Synthesis & Delivery
* Hypothesis testing (H1–H5), measured results verification, and final presentation.

---

## 4. Key Performance Indicators & Acceptance Criteria

| Metric | Target Baseline | EvidenceGuard Target |
| :--- | :--- | :--- |
| **False Answer Rate on Unanswerable (Cat B)** | > 65% (hallucinates answer) | **< 10%** (abstains reliably) |
| **Noise Robustness on Distractors (Cat C)** | < 50% accuracy | **> 85%** accuracy / clean filtering |
| **Citation Faithfulness (Claim-level NLI)** | < 60% supported claims | **> 90%** supported claims |
| **Conflict Detection Precision (Cat E)** | < 30% (silently picks one) | **> 80%** (explicitly flags conflict) |
| **Answerable Question Accuracy (Cat A)** | > 75% | **> 85%** (does not over-abstain) |
| **Selective Risk at 80% Coverage** | > 25% error rate | **< 8%** error rate |
