# EvidenceGuard: Comprehensive Literature Review on Hallucination-Resistant RAG

**Project:** EvidenceGuard RAG  
**Domain:** Information Retrieval, Natural Language Processing, Reliable Question Answering  
**Date:** November 2026  

---

## 1. Introduction & Problem Taxonomy

Retrieval-Augmented Generation (RAG) was introduced to alleviate the parametric knowledge limitations of Large Language Models (LLMs)—specifically, factual hallucination, knowledge obsolescence, and lack of verifiable citations (Lewis et al., 2020; Guu et al., 2020). However, standard naive RAG pipelines suffer from a fundamental architectural flaw: **unconditional generation conditioned on unverified retrieval**.

When an upstream retriever fails (by fetching irrelevant, incomplete, or contradictory passages), downstream LLMs exhibit known failure modes:
1. **Sycophancy & Confabulation:** Generative models often generate answers even when the retrieved context is completely devoid of facts relevant to the query (Zhang et al., 2023; Chen et al., 2024 - RGB Benchmark).
2. **Distractor Sensitivity:** The presence of topically related but non-evidentiary passages ("distractor documents") significantly drops response accuracy, causing models to hallucinate plausible facts (Yuan et al., 2024 - RAGTruth).
3. **Silent Conflict Adoption:** When retrieved passages from different dates or sources present contradictory claims, LLMs typically pick one arbitrarily or merge them into an incoherent, counterfactual blend (Chen et al., 2024).
4. **Citation Fabrication:** LLMs frequently generate citations to chunk IDs or document names that do not contain the asserted claim (Gao et al., 2023).

EvidenceGuard addresses these challenges by replacing unconditional generation with a **calibrated evidence evaluation, conflict detection, and claim verification pipeline**.

---

## 2. Foundations of Traditional and Advanced RAG

### 2.1 Naive RAG vs. Advanced RAG vs. Modular RAG
* **Naive RAG (Lewis et al., 2020):** Follows a linear `Retrieve -> Read` pipeline. The query is embedded via a bi-encoder, top-$k$ nearest neighbors are fetched via Maximum Inner Product Search (MIPS), and concatenated verbatim into the system prompt.
  - *Failure Modes:* Poor precision on multi-hop questions, high sensitivity to chunk boundary fragmentation, and zero abstention capability.
* **Advanced RAG (Gao et al., 2023):** Adds pre-retrieval techniques (query rewriting, query expansion) and post-retrieval techniques (cross-encoder reranking, context selection, prompt compression).
  - *Advantage:* Substantially increases retrieval Recall@k and Precision@k.
* **Modular RAG (Li et al., 2024):** Decouples the system into distinct, interoperable modules: search, memory, routing, predict, evaluate, and verify. EvidenceGuard builds upon the Modular RAG paradigm, utilizing a LangGraph state machine to coordinate modular evaluation before output emission.

### 2.2 Dense Retrieval vs. Sparse Retrieval vs. Hybrid Fusion
* **Dense Bi-Encoder Retrieval (Karpukhin et al., 2020 - DPR):** Projects queries and documents into a shared continuous semantic vector space $\mathbb{R}^d$ using dual encoders:
  $$\text{sim}(q, d) = \langle E_Q(q), E_D(d) \rangle$$
  - *Strength:* Captures semantic synonyms, paraphrases, and conceptual affinity.
  - *Weakness:* Vulnerable to lexical mismatches on domain-specific identifiers, serial numbers, part numbers, and rare entities; prone to "semantic drift" where topically related but factually irrelevant text receives high cosine similarity.
* **Sparse Lexical Retrieval (Robertson & Zaragoza, 2009 - BM25):** Calculates term frequency and inverse document frequency with document length normalization:
  $$\text{score}(D, Q) = \sum_{i=1}^{n} \text{IDF}(q_i) \cdot \frac{f(q_i, D) \cdot (k_1 + 1)}{f(q_i, D) + k_1 \cdot (1 - b + b \cdot \frac{|D|}{\text{avgdl}})}$$
  - *Strength:* Immune to semantic drift; excels at exact identifier, code, and named entity retrieval.
  - *Weakness:* Vocabulary mismatch problem when the user uses synonyms not present in the corpus.
* **Reciprocal Rank Fusion (RRF) (Cormack et al., 2009):** Combines rank positions from multiple distinct retrieval systems without requiring score normalization:
  $$\text{RRF\_Score}(d \in D) = \sum_{m \in M} \frac{1}{k + r_m(d)}$$
  where $M$ is the set of retrievers, $r_m(d)$ is the rank of document $d$ in system $m$, and $k$ is a smoothing constant (standard $k=60$).
  - *Empirical Justification:* Studies show RRF consistently outperforms raw score interpolation ($\alpha \cdot \text{Dense} + (1-\alpha) \cdot \text{BM25}$) because dense cosine scores and BM25 scores follow fundamentally different, non-stationary probability distributions.

---

## 3. Retrieval Improvement & Query Engineering

### 3.1 Cross-Encoder Reranking
* **Mechanism (Nogueira & Cho, 2019):** Unlike bi-encoders that encode query and passage independently into fixed vectors, cross-encoders pass the concatenated pair $(q, d)$ through full cross-attention layers in a transformer:
  $$s(q, d) = \text{Transformer}([CLS] \circ q \circ [SEP] \circ d \circ [SEP])$$
* **Impact on RAG:** Cross-attention allows every token of the query to attend to every token of the passage, capturing subtle grammatical dependencies, negations, and exact qualifiers that bi-encoders compress away.
* **Models:** `BAAI/bge-reranker-large` and `cross-encoder/ms-marco-MiniLM-L-6-v2`. Reranking top-25 bi-encoder candidates down to top-5 delivers an average 18–35% boost in MRR@10 and nDCG@10.

### 3.2 Query Transformation: Rewriting, Expansion, and HyDE
* **Query Rewriting (Ma et al., 2023):** Users frequently submit conversational, ambiguous, or poorly phrased queries. An upstream query rewriting module strips conversational filler and clarifies intent.
* **Hypothetical Document Embeddings - HyDE (Gao et al., 2022):** Prompts an LLM to generate a hypothetical passage that would answer the query, then embeds the hypothetical passage to search the corpus.
  - *Risk:* In unanswerable or counterfactual scenarios, HyDE generates a hallucinated hypothetical document, which then biases the vector search toward distractor passages that match the hallucination!
  - *EvidenceGuard Decision:* We employ multi-query expansion and keyword extraction for retrieval, while deliberately avoiding unconstrained HyDE on unanswerable test suites to prevent hallucination amplification.

---

## 4. Hallucination Resistance & Epistemic Abstention

### 4.1 Benchmark Insights: RGB and RAGTruth
* **RGB Benchmark (Chen et al., AAAI 2024 - arXiv:2309.01431):** Evaluated LLMs across 4 core capabilities:
  1. *Noise Robustness:* Extracting gold answers amid irrelevant passages.
  2. *Negative Rejection:* Refusing to answer when retrieved documents lack the necessary information.
  3. *Information Integration:* Synthesizing multi-document evidence.
  4. *Counterfactual Robustness:* Resisting fabricated or incorrect facts in context.
  - *Key Finding:* State-of-the-art LLMs (including GPT-4) fail dramatically at **Negative Rejection** (often answering with false certainty on unanswerable queries) and **Counterfactual Robustness**.
* **RAGTruth (Yuan et al., ACL 2024 - arXiv:2401.00396):** Annotated ~18,000 naturally generated RAG responses at the word and sentence level across QA, summarization, and data-to-text.
  - *Key Finding:* Hallucination occurs even when relevant context is provided, due to parametric knowledge interference and extrapolation beyond retrieved evidence. Fine-tuning smaller models specifically on grounded hallucination detection matches or exceeds GPT-4 zero-shot prompt evaluation.

### 4.2 Architectural Breakthroughs: Self-RAG and CRAG
* **Self-RAG (Asai et al., ICLR 2024 - arXiv:2310.11511):** Trains an LLM to output special reflection tokens:
  - `[Retrieval]` (decides if retrieval is necessary)
  - `[IsRel]` (evaluates passage relevance: relevant vs. irrelevant)
  - `[IsSup]` (evaluates claim support: fully supported, partially supported, unsupported)
  - `[IsUse]` (evaluates utility score 1–5)
  - *Limitation:* Requires fine-tuning custom base models with special token vocabularies; does not decouple epistemic sufficiency routing from generative decoding.
* **Corrective RAG - CRAG (Yan et al., 2024 - arXiv:2401.15884):** Integrates a lightweight retrieval evaluator returning confidence scores:
  - High confidence $\rightarrow$ Proceed to generation.
  - Low confidence $\rightarrow$ Discard static retrieval, trigger fallback web search.
  - Ambiguous $\rightarrow$ Combine internal documents with decomposed web snippets.
  - *EvidenceGuard Synthesis:* EvidenceGuard adapts CRAG's evaluate-and-route philosophy into a 5-way decision router (`Sufficient`, `Partial`, `Insufficient`, `Conflicting`, `Irrelevant`) executed prior to generation.

### 4.3 Calibrated Selective Generation and Conformal Abstention
* **Selective Prediction (Geifman & El-Yaniv, 2017):** A selective classifier $(f, g)$ outputs prediction $f(x)$ only when confidence $g(x) \ge \theta$, abstaining otherwise.
* **Risk vs. Coverage Trade-off:**
  $$\text{Coverage}(\theta) = \mathbb{E}[\mathbb{I}(g(x) \ge \theta)]$$
  $$\text{Selective Risk}(\theta) = \frac{\mathbb{E}[\mathcal{L}(f(x), y) \cdot \mathbb{I}(g(x) \ge \theta)]}{\text{Coverage}(\theta)}$$
* In EvidenceGuard, the abstention mechanism is not merely an LLM prompt saying "please say you don't know". Instead, it is an empirical classifier calibrated using isotonic regression or Platt scaling over multi-modal retrieval features.

---

## 5. Fine-Tuning Paradigms for Noisy RAG: RAFT and Finetune-RAG

### 5.1 Retrieval Augmented Fine-Tuning (RAFT)
* **RAFT (Zhang et al., UC Berkeley 2024 - arXiv:2403.10131):**
  - Standard fine-tuning trains models in a "closed-book" setting; standard RAG evaluates models in an "open-book" setting.
  - RAFT bridges this divide by training models on questions accompanied by a mixture of **gold documents** and **distractor documents** ($P\%$ gold + $(1-P)\%$ distractors).
  - The model is trained to output:
    1. Verbatim quote from the gold document.
    2. Step-by-step reasoning chain referencing the quote.
    3. Final concise answer.
  - *Result:* When tested on noisy retrieval contexts, RAFT models ignore distractors and anchor their reasoning exclusively to verifiable textual evidence.

### 5.2 Finetuned RAG for Abstention & Noise Resistance
* Recent research (e.g., Lin et al., 2024; Saad-Falcon et al., 2024) demonstrates that models fine-tuned with explicitly unanswerable samples learn to output standardized abstention tokens (e.g., `"INSUFFICIENT_EVIDENCE: The provided context does not state..."`) rather than hallucinating plausibly sounding answers.
* EvidenceGuard incorporates this by fine-tuning on noisy retrieval contexts where distractors are present and negative rejection is rewarded.

---

## 6. Claim-Level Verification and Citation Grounding

### 6.1 Atomic Claim Decomposition
* High-level responses often contain a mixture of supported facts and unsupported hallucinations.
* EvidenceGuard decomposes generated answers into atomic factual claims $\{c_1, c_2, \dots, c_m\}$ using syntactic dependency parsing or LLM extraction.

### 6.2 NLI-Based Grounding Verification
* For each claim $c_i$ and candidate context chunk $p_j$, a Natural Language Inference (NLI) model computes:
  $$P(\text{Entailment} \mid p_j, c_i), \quad P(\text{Neutral} \mid p_j, c_i), \quad P(\text{Contradiction} \mid p_j, c_i)$$
* A claim is certified if and only if $\max_j P(\text{Entailment} \mid p_j, c_i) > \tau_{\text{entail}}$.
* If any claim exhibits $P(\text{Contradiction}) > \tau_{\text{contra}}$, it is flagged immediately for conflict warning or regeneration.

---

## 7. Automated Evaluation Frameworks: ARES and RAGAS

* **RAGAS (Es et al., 2023):** Evaluates RAG pipelines across 4 core metrics:
  - *Faithfulness:* Ratio of claims in generated answer inferable from context.
  - *Answer Relevance:* Semantic alignment between answer and query.
  - *Context Precision:* Signal-to-noise ratio in retrieved context.
  - *Context Recall:* Extent to which retrieved context contains gold answer facts.
* **ARES (Saad-Falcon et al., NAACL 2024 - arXiv:2311.09476):**
  - Employs synthetic training data to train lightweight LM judges for context relevance, faithfulness, and answer relevance.
  - Integrates **Prediction-Powered Inference (PPI)** to combine small human-annotated sets with automated judge scores to construct statistically valid confidence intervals.
  - *EvidenceGuard Stance:* We implement deterministic NLI checks, reference-based lexical metrics, and calibrated LLM judges with PPI-inspired statistical validation to prevent metric gaming.
