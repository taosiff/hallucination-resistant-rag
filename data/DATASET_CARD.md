# Dataset Card: EvidenceGuard Hallucination-Resistance Benchmark

## Dataset Summary
The **EvidenceGuard Benchmark** is a curated, multi-category dataset designed to rigorously assess the vulnerability of Retrieval-Augmented Generation (RAG) systems to hallucinations, ungrounded extrapolations, and false confidence under noisy, incomplete, and contradictory retrieval scenarios.

## Supported Tasks
- **Epistemic Answerability Classification:** Predicting whether retrieved context contains sufficient evidence to answer the query.
- **Evidence Sufficiency Classification (5-Class):** `SUFFICIENT`, `PARTIALLY_SUFFICIENT`, `INSUFFICIENT`, `CONFLICTING`, `IRRELEVANT`.
- **Grounded Answer Generation & Calibrated Abstention:** Generating answers only when supported, qualifying partial answers, and abstaining when evidence is missing.
- **Claim-Level Grounding & Citation Verification:** Sentence-level natural language inference entailment scoring against cited passages.

## Dataset Structure
Each sample adheres to `data/benchmark_schema.json`:
- `id` (str): e.g. `EG-CAT-E-012`
- `category` (enum A-H)
- `category_name` (str)
- `question` (str)
- `passages` (list of passage objects with `id`, `doc_id`, `title`, `text`, `is_gold`, `date`, `source`)
- `ground_truth_answer` (str)
- `expected_decision` (enum 5 states)
- `supporting_passage_ids` (list[str])
- `contradictory_passage_ids` (list[str])
- `is_answerable` (bool)
- `claim_breakdown` (list of atomic claims with gold support booleans)
- `source_metadata` (dict of provenance and curation details)

## Data Sources & Provenance
Adapted and constructed from verified public research corpora:
1. **Natural Questions (NQ) & SQuAD 2.0:** Public domain / CC BY-SA 4.0; used for Categories A (answerable) and B (unanswerable questions designed by adversarial annotators to resemble answerable questions).
2. **HotpotQA:** CC BY-SA 4.0; used for Category F (multi-hop reasoning across multiple Wikipedia articles).
3. **RGB Benchmark:** MIT License; used for Categories C (irrelevant distractor passages), E (conflicting evidence), and H (counterfactual / misleading context).
4. **FEVER (Fact Extraction and VERification):** CC BY-SA 3.0; used for claim-level verification grounding.

## Data Splits
- `train.json`: 120 samples balanced across all categories
- `val.json`: 40 samples balanced across all categories
- `test.json`: 40 samples strictly held-out across all categories
- **Total Benchmark Corpus:** 200 fully annotated samples with verified passages, passage IDs, dates, and atomic claim breakdowns.

## Annotations & Limitations
- All unanswerable and distractor samples have been verified to ensure no hidden answer exists in the distractor passages.
- Conflicting passage pairs have verified contradictory assertions (e.g., differing dates, differing founders, or conflicting scientific constants).
- Temporal changes (Category G) explicitly include document dates so the model has the requisite signal to reason about recency.
