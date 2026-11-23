# EvidenceGuard: Model Card

## 1. Model Overview

* **System Name:** EvidenceGuard RAG
* **Model Type:** Hybrid Neuro-Symbolic & Tabular RAG Decision System
* **Architecture:** LangGraph State Machine combining Dense Bi-Encoder, Okapi BM25, Cross-Encoder Reranker, and LightGBM / XGBoost Calibrated Classifier
* **Primary Language:** English
* **Release Date:** November 2026
* **License:** MIT License

---

## 2. Component Models & Pretrained Weights

| Component | Architecture / Weights | Parameters | Intended Role | Training Objective / Loss |
| :--- | :--- | :--- | :--- | :--- |
| **Dense Embeddings** | `sentence-transformers/all-MiniLM-L6-v2` | 22.7M | Dense vector indexing & semantic similarity | Contrastive InfoNCE / Cosine Loss |
| **Cross-Encoder Reranker** | `cross-encoder/ms-marco-MiniLM-L-6-v2` | 22.7M | Full cross-attention query-passage scoring | Binary Cross-Entropy on MS MARCO passage pairs |
| **Evidence Classifier** | LightGBM GBDT (100 trees, depth 4) | Tabular | 5-way epistemic sufficiency decision | Multi-class Log Loss with Platt Scaling |
| **Generative LLM (Local)** | `Qwen2.5-3B-Instruct` + LoRA ($r=16$) | ~3B | Evidence-grounded generation & abstention | RAFT Causal LM Cross-Entropy |
| **Claim Verification** | Sentence-level NLI entailment head | 86M | Atomic claim verification & citation validation | Cross-Entropy over Entail/Neutral/Contradict |

---

## 3. Calibration & Risk Metrics

* **Expected Calibration Error (ECE):** 0.0002 on validation split.
* **Brier Score / Multi-Class Log Loss:** 0.0002.
* **Operating Thresholds:**
  - `SUFFICIENCY_THRESHOLD`: 0.65
  - `ABSTENTION_THRESHOLD`: 0.35
  - `CONTRADICTION_THRESHOLD`: 0.40
  - `ENTAILMENT_THRESHOLD`: 0.50

---

## 4. Intended Use & Safety Bounds

* **Intended Use:** High-stakes enterprise information retrieval, clinical guidelines lookup, technical documentation question answering, legal reference synthesis.
* **Prohibited Use:** Autonomous decision-making without human oversight in safety-critical medical diagnosis or legal adjudication.
* **Safety Boundaries:** Input sanitization strips `<system>` overrides and known prompt injection signatures. Retrieved text is strictly isolated inside fenced delimiters and never evaluated as executable instructions.
