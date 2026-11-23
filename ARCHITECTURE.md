# EvidenceGuard: System Architecture & Engineering Specification

This document provides a comprehensive technical reference for the **EvidenceGuard RAG** system architecture, data models, state machines, and mathematical formulations.

---

## 1. Architectural Philosophy: Evidence-First Decoupling

Standard RAG architectures couple retrieval directly to text generation:
$$\text{Query } q \longrightarrow \text{Retrieve Top-}k \longrightarrow \text{Prompt } [CLS, \text{Context}, q] \longrightarrow \text{LLM Output}$$

This assumes that retrieval is infallible. EvidenceGuard introduces an explicit **Epistemic Decision Layer** between retrieval and generation:
$$\text{Query } q \longrightarrow \text{Hybrid Retrieve} \longrightarrow \text{Cross-Encoder Rerank} \longrightarrow \mathbf{\text{Evidence Classifier}} \longrightarrow \text{Conditional Route}$$

If and only if the context satisfies mathematical sufficiency criteria does generation proceed.

---

## 2. LangGraph State Machine Specification

EvidenceGuard models execution as a cyclic, checkpointed state machine orchestrated by **LangGraph**.

### 2.1 Graph State Definition (`GraphState`)
The typed state object preserves full auditability across all node transitions:
```python
class GraphState(BaseModel):
    original_query: str
    current_query: str
    rewritten_queries: List[str]
    query_intent: Optional[str]
    entities: List[str]
    retrieved_documents: List[ScoredPassage]
    reranked_documents: List[ScoredPassage]
    evidence_assessment: Optional[EvidenceAssessment]
    epistemic_decision: EpistemicDecision
    confidence_score: float
    conflict_report: Optional[ConflictReport]
    raw_answer: Optional[str]
    final_answer: Optional[str]
    supporting_claims: List[ClaimVerification]
    citations: List[Citation]
    abstention_reason: Optional[str]
    retry_count: int
    max_retries: int
    execution_trace: List[Dict[str, Any]]
```

### 2.2 Node Specifications
1. **`analyze_query`:** Extracts entities, token statistics, and classifies question intent (`factoid`, `explanatory`, `multi_part`).
2. **`rewrite_query`:** Removes conversational filler and expands entity synonyms upon retrieval failure.
3. **`retrieve_documents`:** Executes hybrid search (ChromaDB dense cosine similarity + Rank-BM25) and fuses candidates via Reciprocal Rank Fusion ($k=60$).
4. **`rerank_documents`:** Processes candidate pairs through cross-attention layers of `cross-encoder/ms-marco-MiniLM-L-6-v2`, computing sigmoid-normalized relevance scores.
5. **`assess_evidence`:** Extracts the 12-dimensional feature vector and invokes the calibrated LightGBM classifier.
6. **`detect_conflicts`:** Evaluates top passages for contradictory numerical facts, opposing antonyms, or temporal obsolescence.
7. **`decide_answerability`:** Maps classifier output into one of 5 discrete execution routes.
8. **`generate_answer`:** Synthesizes answer using evidence-grounded prompt tailored to the decision (`SUFFICIENT`, `PARTIALLY_SUFFICIENT`, or `CONFLICTING`).
9. **`verify_claims`:** Deconstructs generated text into atomic sentences; scores NLI entailment against cited passages.
10. **`validate_citations`:** Ensures cited chunk IDs exist in the retrieved corpus and removes fabricated references.
11. **`abstain`:** Constructs a calibrated refusal response explaining why evidence was lacking.
12. **`finalize_response`:** Serializes structured `QueryResponse` with latency telemetry and trace audit.

---

## 3. Mathematical Formulations

### 3.1 Reciprocal Rank Fusion (RRF)
Given ranked candidate lists $R_{\text{dense}}$ and $R_{\text{bm25}}$:
$$\text{RRF}(d) = \sum_{m \in \{\text{dense}, \text{bm25}\}} \frac{1}{60 + r_m(d)}$$

### 3.2 Feature Pipeline for Evidence Assessment
The 12-dimensional vector $\mathbf{x} \in \mathbb{R}^{12}$ comprises:
1. $x_1 = \max_i s_{\text{dense}}(d_i)$ (Top dense score)
2. $x_2 = \max_i s_{\text{rerank}}(d_i)$ (Top cross-encoder score $r_1$)
3. $x_3 = r_1 - r_2$ (Top-1 to Top-2 score margin)
4. $x_4 = r_1 - r_k$ (Top-1 to Top-$k$ decay)
5. $x_5 = \frac{1}{k} \sum_{i=1}^k r_i$ (Reranker mean)
6. $x_6 = \sqrt{\frac{1}{k} \sum_{i=1}^k (r_i - \bar{r})^2}$ (Reranker standard deviation)
7. $x_7 = -\sum_{i=1}^k p_i \log p_i$ (Entropy over softmax reranker distribution)
8. $x_8 = \sum_{i=1}^k \mathbb{I}(r_i \ge 0.50)$ (Passages above threshold)
9. $x_9 = \max_i s_{\text{bm25}}(d_i)$ (Top BM25 score)
10. $x_{10} = |Q_{\text{words}}|$ (Query word count)
11. $x_{11} = \frac{|Q_{\text{tokens}} \cap D_{1,\text{tokens}}|}{|Q_{\text{tokens}}|}$ (Lexical Jaccard overlap)
12. $x_{12} = \text{Conflict}(d_1, d_2) \in [0, 1]$ (Pairwise contradiction signal)

### 3.3 Atomic Claim Entailment Verification
For each claim $c_i$ and cited passage $p_j$:
$$\text{Faithfulness}(c_i) = \mathbb{I}\left(P(\text{Entailment} \mid p_j, c_i) \ge \tau_{\text{entail}}\right) \cdot \mathbb{I}\left(P(\text{Contradiction} \mid p_j, c_i) < \tau_{\text{contra}}\right)$$
where $\tau_{\text{entail}} = 0.50$ and $\tau_{\text{contra}} = 0.40$.
