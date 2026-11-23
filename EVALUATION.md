# EvidenceGuard: Evaluation Framework & Benchmark Methodology

## 1. Evaluation Dimensions

To prevent metric gaming and ensure reproducible statistical rigor, EvidenceGuard evaluates systems across five independent dimensions:

### 1.1 Retrieval Quality Metrics
* **Recall@k:** Fraction of gold supporting passages retrieved in top-$k$.
  $$\text{Recall@k} = \frac{|R_k \cap G|}{|G|}$$
* **Precision@k:** Fraction of top-$k$ retrieved passages that are gold evidence.
* **Mean Reciprocal Rank (MRR):** Reciprocal rank of the first relevant gold passage.
  $$\text{MRR} = \frac{1}{|Q|} \sum_{i=1}^{|Q|} \frac{1}{\text{rank}_i}$$
* **Normalized Discounted Cumulative Gain (nDCG@k):** Measures graded relevance discounting lower-ranked items.

### 1.2 Answer Quality & Correctness Metrics
* **Token-Level F1:** Harmonic mean of precision and recall between prediction tokens and ground truth tokens.
* **Exact Match (EM):** Binary indicator of exact normalized string equivalence.

### 1.3 Faithfulness & Citation Correctness Metrics
* **Claim Support Rate:** Fraction of atomic claims in the generated response that are entailed by cited passages ($P(\text{Entailment}) \ge 0.50$).
* **Citation Precision:** Fraction of cited document IDs that genuinely support the associated claim.
* **Citation Hallucination Rate:** Frequency with which non-existent chunk IDs or ungrounded assertions are generated.

### 1.4 Epistemic Abstention & Selective Risk Metrics
* **Abstention F1:** Harmonic mean of precision and recall on unanswerable and irrelevant queries.
* **False Answer Rate (FAR):** Rate at which the system generates a factual assertion when the context contains zero evidence:
  $$\text{FAR} = \frac{\text{Answered Unanswerable Queries}}{\text{Total Unanswerable Queries}}$$
* **Selective Risk ($\text{SR}(\theta)$):** Error rate among accepted answers when confidence exceeds threshold $\theta$:
  $$\text{SR}(\theta) = \frac{\sum_{i=1}^N \mathcal{L}(f(x_i), y_i) \cdot \mathbb{I}(g(x_i) \ge \theta)}{\sum_{i=1}^N \mathbb{I}(g(x_i) \ge \theta)}$$
* **Coverage ($\text{Cov}(\theta)$):** Fraction of queries answered at threshold $\theta$:
  $$\text{Cov}(\theta) = \frac{1}{N} \sum_{i=1}^N \mathbb{I}(g(x_i) \ge \theta)$$

### 1.5 Conflict Handling Metrics
* **Conflict Detection Recall:** Percentage of contradictory passage pairs flagged.
* **Conflict Detection Precision:** Percentage of flagged conflicts that represent genuine contradictions.
