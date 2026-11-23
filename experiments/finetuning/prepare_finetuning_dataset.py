"""
EvidenceGuard RAFT-Style Fine-Tuning Dataset Generator
Prepares instruction-tuning examples combining gold context with distractors,
teaching models verbatim citation extraction, distractor rejection, and explicit abstention.
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Any

def create_raft_training_examples(benchmark_path: str) -> List[Dict[str, Any]]:
    with open(benchmark_path, "r", encoding="utf-8") as f:
        samples = json.load(f)
        
    finetune_data = []
    
    for s in samples:
        q = s["question"]
        cat = s["category"]
        passages = s["passages"]
        
        # Build context string
        ctx_lines = []
        for p in passages:
            ctx_lines.append(f"Passage [{p['id']}] (Title: {p['title']}): {p['text']}")
        context_str = "\n\n".join(ctx_lines)
        
        # Format target output with CoT reasoning and citation
        if s["expected_decision"] == "SUFFICIENT":
            target = (
                f"Evidence Decision: SUFFICIENT\n"
                f"Reasoning: The retrieved passage [{s['supporting_passage_ids'][0]}] explicitly substantiates the query.\n"
                f"Answer: {s['ground_truth_answer']}"
            )
        elif s["expected_decision"] == "PARTIALLY_SUFFICIENT":
            target = (
                f"Evidence Decision: PARTIALLY_SUFFICIENT\n"
                f"Reasoning: Some sub-claims are grounded in passage [{s['supporting_passage_ids'][0]}], but other query requirements are missing from the text.\n"
                f"Answer: {s['ground_truth_answer']}"
            )
        elif s["expected_decision"] == "CONFLICTING":
            target = (
                f"Evidence Decision: CONFLICTING\n"
                f"Reasoning: The retrieved documents present conflicting factual assertions across distinct sources.\n"
                f"Answer: {s['ground_truth_answer']}"
            )
        elif s["expected_decision"] == "IRRELEVANT":
            target = (
                f"Evidence Decision: IRRELEVANT\n"
                f"Reasoning: The retrieved passages discuss related topics but do not contain factual evidence to answer the question.\n"
                f"Answer: ABSTAIN: The retrieved context is irrelevant to the query."
            )
        else:  # INSUFFICIENT
            target = (
                f"Evidence Decision: INSUFFICIENT\n"
                f"Reasoning: The knowledge base context completely lacks the information requested by the user.\n"
                f"Answer: ABSTAIN: The provided documents do not contain the answer to this question."
            )
            
        system_prompt = (
            "You are EvidenceGuard RAG. You evaluate retrieved passages for sufficiency, "
            "refuse to hallucinate when evidence is missing, cite exact passage IDs, and identify contradictions."
        )
        user_prompt = f"Context:\n{context_str}\n\nQuestion: {q}"
        
        finetune_data.append({
            "id": s["id"],
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": target}
            ]
        })
        
    return finetune_data

def main():
    os.makedirs("experiments/finetuning", exist_ok=True)
    train_ft = create_raft_training_examples("data/splits/train.json")
    val_ft = create_raft_training_examples("data/splits/val.json")
    
    with open("experiments/finetuning/raft_train.jsonl", "w", encoding="utf-8") as f:
        for ex in train_ft:
            f.write(json.dumps(ex) + "\n")
            
    with open("experiments/finetuning/raft_val.jsonl", "w", encoding="utf-8") as f:
        for ex in val_ft:
            f.write(json.dumps(ex) + "\n")
            
    print(f"Generated {len(train_ft)} RAFT training samples and {len(val_ft)} validation samples.")
    print("Saved to experiments/finetuning/raft_train.jsonl and raft_val.jsonl")

if __name__ == "__main__":
    main()
