# EvidenceGuard: Comparative Analysis of Key Research Papers

This document provides a systematic, rigorous comparison of the foundational papers informing the architecture of **EvidenceGuard**.

---

## 1. Master Paper Comparison Matrix

| Paper / Framework | Venue / Date | Core Method | How Imperfect Retrieval & Noise Are Handled | Abstention Mechanism | Primary Limitations | Relevance to EvidenceGuard |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Self-RAG** (Asai et al.) | ICLR 2024 | Instruction-tuning with special reflection tokens (`[Retrieval]`, `[IsRel]`, `[IsSup]`, `[IsUse]`) | Evaluates passage relevance on the fly; discards irrelevant segments before generation | Emits `[IsSup:No]` or refuses to retrieve when facts are absent | Requires training custom token vocabularies; inference decoding latency is 3–4x higher | Directly informs EvidenceGuard's post-retrieval relevance and claim support checks |
| **CRAG (Corrective RAG)** (Yan et al.) | arXiv Jan 2024 (ACL 2024) | Evaluates retrieval confidence with a lightweight evaluator; routes to Correct, Incorrect, or Ambiguous | If low confidence, discards static retrieval and invokes external web search; decomposes and filters chunks | Does not focus on abstention; assumes web search will find truth | Assumes external search is always accessible and safe from web hallucination/poisoning | Provides the blueprint for our 5-way decision router prior to LLM generation |
| **RAFT** (Zhang et al.) | UC Berkeley / arXiv Mar 2024 | Retrieval-Augmented Fine-Tuning; trains model with mixed gold and distractor documents | Explicitly trains the LLM to ignore distractors and cite verbatim quotes in CoT reasoning | Can be trained to output explicit refusal when only distractors are present | Requires full domain fine-tuning dataset; does not calibrate prediction confidence | Informs our Phase 7 fine-tuning recipe: training models to isolate verbatim citations amid noisy distractors |
| **RGB Benchmark** (Chen et al.) | AAAI 2024 (arXiv Sep 2023) | Benchmark evaluating Noise Robustness, Negative Rejection, Info Integration, Counterfactual Robustness | Tests 4 distinct noise conditions; exposes that LLMs fail severely on negative rejection | Evaluates models' ability to output rejection string (e.g., "I cannot answer") | Diagnostic benchmark only; does not provide an architectural mitigation solution | Provides the exact taxonomy for EvidenceGuard's Category A–H evaluation benchmark |
| **RAGTruth** (Yuan et al.) | ACL 2024 (arXiv Jan 2024) | Word-level and sentence-level hallucination corpus (~18k responses) across QA, summary, data-to-text | Distinguishes between context-grounded errors vs. parametric knowledge intrusions | Trained smaller classifier models on annotated corpus for hallucination detection | Static dataset; covers limited open-domain QA domains | Justifies using lightweight, fine-tuned classifiers over expensive GPT-4 prompt checks |
| **ARES** (Saad-Falcon et al.) | NAACL 2024 (arXiv Nov 2023) | Automated evaluation framework using synthetic data, fine-tuned lightweight judges, and PPI | Uses hard negative synthetic samples to evaluate context relevance and answer faithfulness | Measures faithfulness drop under noisy passages | Synthetic data generation can introduce subtle judge biases | Guides our Phase 9 automated evaluation pipeline and statistical confidence estimation |
| **RAGAS** (Es et al.) | EACL 2024 | Reference-free metrics: Faithfulness, Answer Relevance, Context Precision, Context Recall | Measures overlap between retrieved context and generated claims | Low faithfulness score flags unsupported generations | Relies purely on LLM-as-a-judge prompts; susceptible to judge hallucinations and prompt sensitivity | Implemented as one component alongside deterministic NLI checks |
| **SQuAD 2.0** (Rajpurkar et al.) | ACL 2018 | Reading comprehension with over 50,000 unanswerable questions designed to resemble answerable ones | Evaluates model ability to detect when reading context lacks plausible answer | Explicit "no-answer" classification head / abstention threshold | Extractive span-based reading comprehension rather than generative RAG | Core foundation for training and benchmarking negative rejection on plausible distractors |
| **Conformal Prediction for LLMs** (Quach et al., 2023) | NeurIPS 2023 | Distribution-free uncertainty quantification; guarantees error rate bounds on answer sets | Calibrates confidence thresholds to bound selective risk at specified user error rate $\alpha$ | Abstains whenever the prediction set size is null or exceeds threshold | Requires exchangeable calibration dataset; conservative thresholds may reduce coverage | Informs our Risk-Coverage calibration curves in Phase 5 |

---

## 2. Deep Dive: Architectural Insights Applied to EvidenceGuard

### 2.1 The Failure Mode of Prompt-Only Abstention
Multiple empirical studies (RGB, RAGTruth, Chen et al.) demonstrate that instructing an LLM in the system prompt:
> *"If the context does not contain enough information, reply with 'I don't know'."*

is fundamentally insufficient:
1. **Instruction Degradation Under Long Contexts:** When context length exceeds 2,000 tokens, LLMs experience "lost-in-the-middle" attention degradation (Liu et al., 2023), leading to forgotten negative constraints.
2. **Parametric Leakage:** Pretrained LLMs have strong internal priors; if asked about a famous entity, they will answer from parametric memory even if the prompt explicitly forbids it.
3. **Plausibility Biases:** LLMs are autoregressive next-token predictors trained to minimize cross-entropy loss, naturally favoring fluent plausible completions over abrupt abstentions.

**EvidenceGuard Solution:** Decouple the decision to answer from the generation process. An external, calibrated **Evidence Sufficiency Classifier** determines whether generation is mathematically permitted before the LLM receives the prompt.

### 2.2 The Synergy of Hybrid Search + Cross-Encoder Reranking
Dense bi-encoder retrieval alone achieves ~65–75% Recall@5 on complex queries because dense embeddings suffer from semantic drift when queries contain rare technical jargon, specific numbers, or exact codes. Conversely, BM25 fails when queries utilize synonyms or colloquial phrasing.

By combining:
$$\text{Candidate Pool} = \text{Dense}_{\text{top-}50} \cup \text{BM25}_{\text{top-}50}$$
via Reciprocal Rank Fusion (RRF), EvidenceGuard captures both semantic intent and exact lexical terms. Feeding these candidates into a Cross-Encoder (`bge-reranker-base`) enables full token-to-token cross-attention, reordering candidates with an empirical 25–35% increase in MRR@5.

### 2.3 The Necessity of Claim-Level Verification
Document-level citations (e.g., `[Document 1]`) are frequently inaccurate. A 150-word answer might contain six distinct factual statements; three may be supported by paragraph 2 of Document 1, one might be partially supported by Document 3, and two might be complete hallucinations introduced by the model.

EvidenceGuard adopts an **Atomic Claim Verification** stage:
1. Sentence splitting and claim extraction.
2. Pairwise NLI scoring against retrieved passages.
3. Citation binding at the passage level, not document level.
4. Suppressing or flagging claims where $P(\text{Entailment}) < \tau$.
