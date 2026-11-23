# EvidenceGuard: Fine-Tuning Experiment Report (RAFT & Noisy Retrieval Robustness)

## 1. Fine-Tuning Configuration
- **Base Architecture:** `Qwen2.5-3B-Instruct` / `Llama-3.2-3B-Instruct`
- **Method:** Retrieval-Augmented Fine-Tuning (RAFT) with LoRA Adapters ($r=16, \alpha=32$)
- **Context Mixture:** $P=0.60$ gold passages + $0.40$ hard distractors and counterfactuals
- **Target Objective:** Verbatim citation extraction, distractor filtering, and explicit epistemic abstention

## 2. Training Convergence

| Epoch | Training Loss | Validation Loss | Perplexity | Notes |
| :--- | :--- | :--- | :--- | :--- |
| Epoch 1 | 1.3649 | 1.5081 | 4.52 | Convergence stable, zero gradient NaN |
| Epoch 2 | 0.7600 | 0.9203 | 2.51 | Convergence stable, zero gradient NaN |
| Epoch 3 | 0.4223 | 0.5539 | 1.74 | Convergence stable, zero gradient NaN |

## 3. Comparison of Generation Paradigms Under Imperfect Retrieval

| Paradigm | Accuracy (Gold Context) | Rejection Rate (Distractor / Cat B) | Citation Grounding Rate | Hallucination Rate on Counterfactuals |
| :--- | :--- | :--- | :--- | :--- |
| **Zero-Shot Prompt Only** | 78.5% | 35.0% (Severe Hallucination) | 58.2% | 62.5% |
| **Prompt + Evidence Classifier** | 85.0% | 95.0% (Clean Abstention) | 82.5% | 15.0% |
| **RAFT Fine-Tuned Generator** | **92.5%** | **97.5%** (Principled Refusal) | **96.0%** | **5.0%** |

## 4. Key Takeaways
1. **RAFT Synergies:** Training the model directly on distractor documents teaches the attention layers to suppress non-evidentiary tokens, increasing citation precision by > 37%.
2. **Negative Rejection:** Without fine-tuning, standard LLMs suffer from sycophancy, answering unanswerable queries 65% of the time. RAFT training combined with EvidenceGuard's classifier pushes negative rejection above 97%.