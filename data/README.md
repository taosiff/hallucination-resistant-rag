# EvidenceGuard Benchmark Dataset: Overview & Architecture

## 1. Benchmark Purpose

The EvidenceGuard Evaluation Benchmark is engineered specifically to measure whether a Retrieval-Augmented Generation system can resist hallucinations under imperfect, noisy, and deceptive retrieval conditions.

Unlike conventional QA benchmarks (e.g., SQuAD 1.1 or TriviaQA) which assume that the top retrieved context always contains the correct answer, real-world search corpora frequently yield distracting, outdated, or conflicting texts. The EvidenceGuard benchmark forces the model to perform **epistemic decision-making** before generating an answer.

---

## 2. Category Taxonomy (Categories A through H)

| Category Code | Category Name | Context Characteristics | Expected Epistemic Decision | Target Behavior |
| :--- | :--- | :--- | :--- | :--- |
| **Category A** | **Answerable** | Contains complete, unambiguous evidence directly answering the query. | `SUFFICIENT` | Generate direct, concise answer grounded in context; cite exact passages. |
| **Category B** | **Unanswerable** | Retrieved context is completely devoid of facts needed to answer the question. | `INSUFFICIENT` | Principled abstention: state clearly that the knowledge base lacks necessary information. |
| **Category C** | **Irrelevant Retrieval** | Context is semantically similar (shares topic/keywords) but does not contain the answer. | `IRRELEVANT` | Identify irrelevance; attempt query reformulation or abstain without guessing. |
| **Category D** | **Partial Evidence** | Context answers part of a composite query, but omits critical elements. | `PARTIALLY_SUFFICIENT` | Generate qualified answer: state supported facts with citations, explicitly flag missing parts. |
| **Category E** | **Conflicting Evidence** | Different retrieved passages assert contradictory claims for the same entity/event. | `CONFLICTING` | Identify the contradiction; cite both perspectives; do not silently pick one. |
| **Category F** | **Multi-Hop Evidence** | Answer requires synthesizing disconnected facts spread across 2 or more distinct passages. | `SUFFICIENT` | Synthesize multi-passage chain of reasoning; cite all contributing passages. |
| **Category G** | **Outdated Evidence** | Older document asserts a historical fact superseded by a newer document. | `CONFLICTING` / `PARTIAL` | Recognize temporal discrepancies using publication dates; explain update history. |
| **Category H** | **Misleading Context** | Retrieved passages contain plausible counterfactuals or deceptive distractors. | `INSUFFICIENT` / `IRRELEVANT` | Reject false premise; do not adopt counterfactual assertions into response. |

---

## 3. Split Strategy & Data Leakage Prevention

1. **Zero Overlap of Source Documents:** The split is partitioned at the **document and topic cluster level**, not at the question level. If document $D_k$ appears in the test split, neither $D_k$ nor any paraphrase of $D_k$ is present in train or validation splits.
2. **Deterministic Partitioning:**
   - **Train Split (60%):** Used for fitting the Evidence Sufficiency Classifier and fine-tuning experiments (Phase 7).
   - **Validation Split (20%):** Used for threshold tuning, hyperparameter selection, and probability calibration.
   - **Test Split (20%):** Held out strictly for final un-tuned evaluation. Never exposed to model fitting or prompt tuning.

---

## 4. Benchmark Validation & Quality Control

* **No Automated Unverified Labels:** Every sample in the gold test set has human-verified annotations for `is_answerable`, `expected_decision`, and `supporting_passage_ids`.
* **Deterministic Schema Check:** All JSON files in `data/splits/` are validated against `data/benchmark_schema.json` using JSON Schema validators.
