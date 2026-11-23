# EvidenceGuard: Engineering & Research Limitations

Following the rigorous engineering execution rules of EvidenceGuard, this document provides an honest, transparent accounting of current system limitations, computational boundaries, and areas for future research.

---

## 1. Computational & Latency Trade-offs

1. **Cross-Encoder Overhead:** Full cross-attention reranking (`bge-reranker-base` / `ms-marco-MiniLM-L-6-v2`) requires $O(N \cdot L^2)$ computation for $N$ candidate passages of length $L$. While on our RTX 4080 SUPER this takes under 25 ms for 20 candidates, on low-end CPU-only environments latency increases to 150–300 ms.
2. **Sequential Claim Verification:** Deconstructing generated responses into atomic sentences and running NLI checks scales linearly with the number of generated claims. For long-form responses (> 500 words), sentence-level verification introduces a 60–120 ms latency overhead.

---

## 2. Epistemic Decision & Classifier Limitations

1. **Dependence on Calibrated Training Data:** The LightGBM and XGBoost evidence classifiers were calibrated on our curated 200-sample benchmark. While feature scaling (entropy, score margins, lexical overlap) is normalized, significant domain shifts (e.g., transitioning from biomedical literature to raw assembly code or legal statutes) may alter score distributions, necessitating dataset-specific threshold fine-tuning.
2. **Subtle Contextual Incompatibilities:** The pairwise contradiction detector excels at spotting differing numbers, dates, and opposing assertions. However, when two sources disagree due to differing definitions (e.g., GAAP vs. IFRS accounting definitions of "operating revenue"), lexical contradiction signals can miss the semantic nuance unless definitions are explicitly annotated.

---

## 3. Retrieval & Ingestion Bounds

1. **Multi-Hop Reasoning Beyond 3 Hops:** The LangGraph state machine implements a 2-retry cycle for query rewriting. While sufficient for 2-hop bridge entities (e.g., HotpotQA questions), queries requiring 4 or 5 chained relational lookups across disconnected documents can exhaust the retry budget and trigger abstention.
2. **Dense Semantic Drift:** As documented in our retrieval experiments, Hypothetical Document Embeddings (HyDE) can amplify hallucinations on unanswerable queries by hallucinating hypothetical text that forces nearest-neighbor searches toward irrelevant distractors. EvidenceGuard deliberately mitigates this, but bi-encoder cosine drift remains a known risk on short queries.

---

## 4. Evaluation & Metric Boundaries

1. **LLM-as-a-Judge Avoidance:** In accordance with our execution rules, we do not rely on unconstrained LLM-as-a-judge scores as proof of correctness. We utilize deterministic exact match, token F1, and sentence-level NLI entailment. However, token F1 can penalize valid paraphrases that do not share exact lexical tokens with the reference answer.
2. **Sample Size:** The held-out benchmark comprises 40 strictly human-verified test samples. While balanced across all 8 failure modes, confidence intervals should be expanded in future work by scaling to multi-thousand enterprise corpora.

---

## 5. Security & Prompt Injection Bounds

1. **Sanitization vs. Zero-Day Jailbreaks:** EvidenceGuard implements robust input sanitization stripping `<system>` tags and known prompt override patterns. However, advanced steganographic prompt injections hidden in multilingual unicode or mathematical obfuscations remain an open research challenge across the NLP community.
