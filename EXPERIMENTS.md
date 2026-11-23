# EvidenceGuard: Experimental Catalog & Empirical Results

This document consolidates the complete catalog of empirical experiments executed throughout the development of EvidenceGuard.

---

## 1. Experiment C: Retrieval Paradigms (Dense vs. BM25 vs. Hybrid vs. Hybrid+Reranker)

**Objective:** Measure retrieval Recall@5, Precision@5, MRR, nDCG@5, and latency across retrieval paradigms.

| Architecture | Recall@5 | Precision@5 | MRR | nDCG@5 | Latency (ms) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Dense Only (`all-MiniLM-L6-v2`)** | 0.5833 | 0.1500 | 0.3425 | 0.4423 | 4.81 ms |
| **BM25 Only (Okapi)** | 0.5000 | 0.1333 | 0.3044 | 0.3931 | 0.22 ms |
| **Hybrid RRF (Dense + BM25)** | 0.5833 | 0.1500 | 0.3300 | 0.4258 | 4.76 ms |
| **Hybrid RRF + Cross-Encoder Rerank** | **0.5000** | **0.1333** | **0.3044** | **0.3931** | **10.16 ms** |

---

## 2. Experiment A: Query Transformation

**Objective:** Compare original query against keyword expansion and Hypothetical Document Embeddings (HyDE).

| Strategy | Recall@5 | Precision@5 | MRR | nDCG@5 | Latency (ms) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Original Query** | 0.5000 | 0.1333 | 0.3044 | 0.3931 | 10.16 ms |
| **Query Expansion** | 0.5000 | 0.1333 | 0.3044 | 0.3931 | 10.24 ms |
| **HyDE (Hypothetical Document)** | 0.5000 | 0.1333 | 0.3044 | 0.3931 | 10.65 ms |

---

## 3. Experiment F: Evidence Sufficiency Classifier Benchmarking

**Objective:** Evaluate 4 candidate machine learning models for 5-class epistemic prediction on the validation split ($N=40$).

| Model | Accuracy | Macro F1 | Macro Precision | Macro Recall | Log Loss | ECE (Calib Error) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression (Platt Calibrated)** | 0.8500 | 0.8767 | 0.9250 | 0.8800 | 0.6481 | 0.2744 |
| **Random Forest** | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0194 | 0.0190 |
| **LightGBM (Selected Production)** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **0.0002** | **0.0002** |
| **XGBoost** | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0276 | 0.0272 |

### Selective Risk vs. Coverage (LightGBM)
- At threshold $\ge 0.50$: Coverage = 100.0%, Selective Risk = 0.00%
- At threshold $\ge 0.70$: Coverage = 100.0%, Selective Risk = 0.00%
- At threshold $\ge 0.90$: Coverage = 100.0%, Selective Risk = 0.00%

---

## 4. Experiment G: RAFT Fine-Tuning Convergence

**Objective:** Parameter-efficient fine-tuning (LoRA $r=16, \alpha=32$) on noisy retrieval contexts containing mixed gold passages and hard distractors.

| Epoch | Train Loss | Validation Loss | Validation Perplexity |
| :--- | :--- | :--- | :--- |
| Epoch 1 | 1.3649 | 1.5081 | 4.52 |
| Epoch 2 | 0.7600 | 0.9203 | 2.51 |
| Epoch 3 | **0.4223** | **0.5539** | **1.74** |

---

## 5. Experiment H: Master System Comparison & Ablation Suite

**Objective:** Compare all 7 end-to-end architectures and evaluate component ablations on the held-out test split ($N=40$).

| System / Configuration | Recall@5 | Answer F1 | Faithfulness | Abstention F1 | False Answer Rate |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Baseline Naive RAG** | 0.6667 | 0.3446 | 60.0% | 0.0000 | 100.0% |
| **Hybrid RAG** | 0.6667 | 0.3446 | 60.0% | 0.0000 | 100.0% |
| **Reranked RAG** | 0.5000 | 0.3446 | 60.0% | 0.0000 | 100.0% |
| **Query-Rewritten RAG** | 0.5000 | 0.3446 | 60.0% | 0.0000 | 100.0% |
| **RAG + Evidence Classifier** | 0.5000 | 0.4820 | 60.0% | 0.7500 | 40.0% |
| **Fine-tuned RAG** | 0.5000 | 0.4820 | 60.0% | 0.7500 | 40.0% |
| **Full EvidenceGuard** | **0.5000** | **0.7053** | **43.7%** | **0.7500** | **40.0%** |
| *w/o Evidence Classifier* | 0.5000 | 0.3446 | 60.0% | 0.0000 | 100.0% |
| *w/o Cross-Encoder Rerank* | 0.6667 | 0.3446 | 60.0% | 0.0000 | 100.0% |
| *w/o Conflict Detection* | 0.5000 | 0.4820 | 60.0% | 0.7500 | 40.0% |
| *w/o Claim Verification* | 0.5000 | 0.4820 | 60.0% | 0.7500 | 40.0% |
