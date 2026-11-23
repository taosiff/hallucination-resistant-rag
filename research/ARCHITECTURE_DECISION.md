# EvidenceGuard: Architecture Decision Records (ADR)

This document formalizes every major architectural and algorithmic choice in EvidenceGuard, adhering to the rigorous standard:
- What problem it solves.
- How it works.
- Its computational cost.
- Its limitations.
- Why it is appropriate for this project.

---

## ADR 01: Hybrid Dense-Sparse Retrieval with Reciprocal Rank Fusion (RRF)

### 1. Problem Solved
Pure dense retrieval (bi-encoders) fails on queries containing specific acronyms, product serials, code symbols, and exact numeric constants because embeddings compress detailed token identities into fixed-dimensional vectors. Pure sparse retrieval (BM25) fails when queries use paraphrases, synonyms, or conceptual questions without exact keyword overlap.

### 2. How It Works
* A query $q$ is dispatched concurrently to:
  1. Dense vector retriever (ChromaDB with `BAAI/bge-large-en-v1.5` cosine index), yielding ranked list $R_{\text{dense}}$.
  2. Sparse lexical retriever (Rank-BM25 with token-level Okapi weighting), yielding ranked list $R_{\text{bm25}}$.
* RRF combines the ranked lists:
  $$\text{RRF}(d) = \frac{1}{60 + \text{rank}_{\text{dense}}(d)} + \frac{1}{60 + \text{rank}_{\text{bm25}}(d)}$$
  Top-$K$ candidates (typically $K=20$) are extracted for reranking.

### 3. Computational Cost
* Dense: Vector search in ChromaDB takes $\sim 4-8$ ms for 50,000 chunks.
* Sparse: In-memory BM25 index takes $< 2$ ms.
* RRF merge: $O(K \log K)$ sorting, taking $< 0.5$ ms.
* Total Overhead: $< 12$ ms.

### 4. Limitations
RRF is rank-based rather than score-based; it ignores the absolute magnitude of similarity scores. Thus, an irrelevant document that barely ranks #1 in BM25 because of a single rare word could receive the same rank score as a highly relevant dense result.

### 5. Why Appropriate for EvidenceGuard
EvidenceGuard must withstand out-of-vocabulary technical queries and adversarial distractors. Hybrid RRF ensures that retrieval recall does not collapse when either dense semantic search or sparse keyword search fails.

---

## ADR 02: Cross-Encoder Reranking (`BAAI/bge-reranker-base`)

### 1. Problem Solved
Bi-encoders independently encode query $q$ and passage $p$ without cross-token interaction ($E(q) \cdot E(p)$). This leads to false positives where passages share the general topic but do not answer the specific query relation (e.g., confusing "Who founded X?" with "Who was hired by X?").

### 2. How It Works
The top-20 candidates from hybrid retrieval are concatenated with the query as `[CLS] query [SEP] passage [SEP]` and processed through all transformer cross-attention layers of `BAAI/bge-reranker-base`. The output logit from the `[CLS]` token is passed through a sigmoid function to yield a normalized relevance score $s_i \in [0, 1]$.

### 3. Computational Cost
* Forward pass of 20 pairs on NVIDIA RTX 4080 SUPER:
  - Batch size 20, max length 512 tokens.
  - VRAM: $\sim 1.2$ GB.
  - Latency: $\sim 22$ ms.

### 4. Limitations
Computational complexity is $O(N \cdot (L_q + L_p)^2)$ where $N$ is candidate count. It cannot scale to searching an entire database; it can only rerank a small pre-filtered candidate pool ($N \le 50$).

### 5. Why Appropriate for EvidenceGuard
The cross-encoder score distribution provides the primary statistical signal for the downstream Evidence Sufficiency Classifier. The margin between the top score $s_1$ and runner-up $s_2$ strongly indicates whether a unique evidentiary ground truth exists.

---

## ADR 03: Calibrated Evidence Sufficiency Classifier vs. Prompt-Only Abstention

### 1. Problem Solved
LLMs asked to "abstain if you don't know" exhibit high false-positive generation rates (>60% in RGB benchmark). They suffer from hallucination and sycophancy when passages are semantically related but factually insufficient.

### 2. How It Works
Before invoking any generation LLM, an external classifier assesses the evidence state using a 12-dimensional feature vector extracted from the retrieval and reranking stage:
- Top cross-encoder score $s_1$.
- Score margin $\Delta s = s_1 - s_2$.
- Score entropy $H(s) = -\sum \hat{s}_i \log \hat{s}_i$.
- Top BM25 score and BM25 score decay.
- Query length, entity density, and token overlap ratio.
- Pairwise inter-passage contradiction probability (NLI signal).
- Passage length and coverage of query nouns.

A calibrated classifier (LightGBM / XGBoost with Platt scaling) predicts one of 5 discrete states:
1. `SUFFICIENT`: Top evidence is strong, uncontested, and covers query requirements.
2. `PARTIAL`: Evidence answers part of a multi-part query, but lacks key facts.
3. `CONFLICTING`: High-relevance passages make mutually contradictory factual claims.
4. `INSUFFICIENT`: Passages are completely uninformative for the question.
5. `IRRELEVANT`: Passages are semantic distractors with low reranker confidence.

### 3. Computational Cost
Feature extraction + LightGBM inference: $< 3$ ms on CPU.

### 4. Limitations
Requires a balanced calibration dataset across all 5 classes to fit decision boundaries and probability calibrations.

### 5. Why Appropriate for EvidenceGuard
It shifts the epistemic decision from stochastic autoregressive sampling to a deterministic, inspectable, and auditable classifier. If the classifier predicts `INSUFFICIENT`, the pipeline immediately halts and executes a principled abstention, saving token costs and eliminating hallucination risk.

---

## ADR 04: LangGraph State-Machine Orchestration

### 1. Problem Solved
Linear DAG frameworks (such as basic LangChain chains or sequential pipelines) cannot handle conditional branching, cyclic query reformulations, dynamic retry bounds, and granular state checkpointing.

### 2. How It Works
EvidenceGuard defines a strictly typed `EvidenceGuardState` (using Pydantic v2) and models the workflow as a LangGraph state machine. Edges are governed by conditional routers:
- If `assess_evidence` outputs `IRRELEVANT` and retry count $< 2$, route to `rewrite_query` and re-retrieve.
- If retry count $\ge 2$, route to `abstain`.
- If `assess_evidence` outputs `CONFLICTING`, route to `detect_conflicts` $\rightarrow$ `generate_conflict_answer`.
- If `SUFFICIENT`, route to `generate_answer` $\rightarrow$ `verify_claims` $\rightarrow$ `validate_citations`.

### 3. Computational Cost
Graph state transitions and serialization: $< 1$ ms overhead per node.

### 4. Limitations
Slightly higher development complexity compared to linear chains; state schemas must be maintained strictly.

### 5. Why Appropriate for EvidenceGuard
Provides 100% execution observability. Every node step, score, and transition is inspectable via UI drawers and telemetry (LangSmith/MLflow), allowing researchers and users to debug why the system decided to answer or abstain.

---

## ADR 05: Sentence-Level Atomic Claim Extraction and NLI Grounding

### 1. Problem Solved
Document-level citation (e.g., citing an entire PDF or 1,000-character chunk) disguises sentence-level hallucinations. Generated text may blend real facts with fabricated claims.

### 2. How It Works
1. The generated response is parsed into individual sentences $S = \{s_1, \dots, s_k\}$.
2. Each sentence is stripped of conversational boilerplate to isolate atomic factual claims.
3. An NLI cross-encoder model (`nli-deberta-v3-base` or RoBERTa-NLI) evaluates premise-hypothesis pairs:
   $$\text{Premise} = \text{Cited Passage } p_j, \quad \text{Hypothesis} = \text{Claim } s_i$$
4. Output probabilities $[P_{\text{entail}}, P_{\text{neutral}}, P_{\text{contradict}}]$ are computed:
   - If $P_{\text{contradict}} > 0.4$, claim is flagged as **Contradicted**.
   - If $P_{\text{entail}} < 0.5$, claim is flagged as **Unsupported / Hallucinated**.
   - If $P_{\text{entail}} \ge 0.5$, claim is certified as **Faithful**.
5. The final output appends verified citations directly to supported sentences and redacts or annotates unverified claims.

### 3. Computational Cost
Forward pass of $K$ claims against cited passages: $\sim 25-45$ ms on GPU.

### 4. Limitations
NLI models can occasionally struggle with complex numerical reasoning, units conversion, or temporal math across distant context sentences.

### 5. Why Appropriate for EvidenceGuard
Guarantees claim-level faithfulness. If the LLM slips an ungrounded hallucination into an otherwise good answer, the claim verifier catches it before final presentation.

---

## ADR 06: Vector Database Selection (ChromaDB Primary with In-Memory FAISS)

### 1. Problem Solved
Heavyweight vector databases (Milvus, Pinecone, Qdrant cluster) introduce complex infrastructure, external network hops, and Docker daemon dependencies.

### 2. How It Works
- **ChromaDB:** Runs embedded directly in Python process with SQLite/DuckDB backing on local NVMe disk (`D:\`). Provides metadata filtering, collection isolation, and simple document management.
- **FAISS:** Used in evaluation scripts for microsecond-level vector search benchmarking.

### 3. Computational Cost
Zero network latency; embedded function calls execute in single-digit milliseconds.

### 4. Limitations
ChromaDB embedded is single-node and not built for multi-tenant cloud-scale horizontal scaling (>10 million vectors), but is ideal for local, secure, enterprise-grade corpora under 1,000,000 documents.

### 5. Why Appropriate for EvidenceGuard
Maximum speed, zero external dependencies, robust local file persistence on Windows workstation, and fully reproducible across environments.
