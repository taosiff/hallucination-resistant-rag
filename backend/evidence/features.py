"""
EvidenceGuard Feature Extraction Pipeline for Evidence Assessment
Extracts 12 calibrated distributional, lexical, semantic, and inter-passage signals
from the retrieved and reranked passages.
"""

import math
import re
from typing import List, Dict, Any, Tuple
import numpy as np

from backend.core.models import ScoredPassage

FEATURE_NAMES = [
    "top_dense_score",
    "top_rerank_score",
    "rerank_score_margin_1_2",
    "rerank_score_margin_1_k",
    "rerank_score_mean",
    "rerank_score_std",
    "rerank_entropy",
    "num_passages_above_threshold",
    "top_bm25_score",
    "query_len_words",
    "lexical_overlap_ratio",
    "pairwise_contradiction_signal"
]

def extract_evidence_features(
    query: str,
    passages: List[ScoredPassage],
    rerank_threshold: float = 0.50
) -> Dict[str, float]:
    """
    Extracts a 12-dimensional feature dictionary for evidence sufficiency classification.
    """
    query_tokens = set(re.findall(r"\b\w+\b", query.lower()))
    q_len = len(query_tokens)
    
    if not passages:
        return {
            "top_dense_score": 0.0,
            "top_rerank_score": 0.0,
            "rerank_score_margin_1_2": 0.0,
            "rerank_score_margin_1_k": 0.0,
            "rerank_score_mean": 0.0,
            "rerank_score_std": 0.0,
            "rerank_entropy": 0.0,
            "num_passages_above_threshold": 0.0,
            "top_bm25_score": 0.0,
            "query_len_words": float(q_len),
            "lexical_overlap_ratio": 0.0,
            "pairwise_contradiction_signal": 0.0
        }

    # Dense scores
    dense_scores = [p.dense_score for p in passages if p.dense_score is not None]
    top_dense = max(dense_scores) if dense_scores else 0.0
    
    # BM25 scores
    bm25_scores = [p.bm25_score for p in passages if p.bm25_score is not None]
    top_bm25 = max(bm25_scores) if bm25_scores else 0.0

    # Reranker scores
    rerank_scores = [p.rerank_score for p in passages if p.rerank_score is not None]
    if not rerank_scores:
        rerank_scores = [top_dense] * len(passages)
        
    s_sorted = sorted(rerank_scores, reverse=True)
    r1 = s_sorted[0]
    r2 = s_sorted[1] if len(s_sorted) > 1 else 0.0
    rk = s_sorted[-1] if len(s_sorted) > 0 else 0.0
    
    margin_1_2 = max(0.0, r1 - r2)
    margin_1_k = max(0.0, r1 - rk)
    r_mean = float(np.mean(s_sorted))
    r_std = float(np.std(s_sorted)) if len(s_sorted) > 1 else 0.0
    
    # Entropy of normalized softmax scores
    exp_s = np.exp(s_sorted - np.max(s_sorted))
    probs = exp_s / (np.sum(exp_s) + 1e-12)
    entropy = -float(np.sum(probs * np.log(probs + 1e-12)))
    
    num_above = float(sum(1 for s in s_sorted if s >= rerank_threshold))
    
    # Lexical overlap: token intersection between query and top passage
    top_text = passages[0].text if passages else ""
    p_tokens = set(re.findall(r"\b\w+\b", top_text.lower()))
    overlap_ratio = len(query_tokens & p_tokens) / max(1, q_len)
    
    # Pairwise contradiction signal: check for conflict markers or negation shifts in top passages
    contradiction_signal = 0.0
    if len(passages) >= 2:
        text1 = passages[0].text.lower()
        text2 = passages[1].text.lower()
        # Look for explicit conflicting numeric entities or opposing antonyms
        conflict_tokens = ["however", "dispute", "instead", "whereas", "contradicts", "differed", "demoted", "retired"]
        has_marker = any(m in text1 or m in text2 for m in conflict_tokens)
        
        # Check if numbers/dates differ while sharing nouns
        nums1 = set(re.findall(r"\b\d{4}\b|\b\d{2,}\b", text1))
        nums2 = set(re.findall(r"\b\d{4}\b|\b\d{2,}\b", text2))
        if nums1 and nums2 and (nums1 != nums2) and has_marker:
            contradiction_signal = 0.85
        elif has_marker:
            contradiction_signal = 0.45
        elif (passages[0].date and passages[1].date and passages[0].date != passages[1].date):
            contradiction_signal = 0.35

    return {
        "top_dense_score": float(top_dense),
        "top_rerank_score": float(r1),
        "rerank_score_margin_1_2": float(margin_1_2),
        "rerank_score_margin_1_k": float(margin_1_k),
        "rerank_score_mean": float(r_mean),
        "rerank_score_std": float(r_std),
        "rerank_entropy": float(entropy),
        "num_passages_above_threshold": float(num_above),
        "top_bm25_score": float(top_bm25),
        "query_len_words": float(q_len),
        "lexical_overlap_ratio": float(overlap_ratio),
        "pairwise_contradiction_signal": float(contradiction_signal)
    }

def feature_dict_to_array(fdict: Dict[str, float]) -> np.ndarray:
    return np.array([fdict[k] for k in FEATURE_NAMES], dtype=np.float32)
