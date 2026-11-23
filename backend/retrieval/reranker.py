"""
EvidenceGuard Cross-Encoder Reranker
Evaluates token-level cross-attention relevance and normalizes logits to calibrated probabilities.
"""

from typing import List
import torch
import numpy as np
from sentence_transformers import CrossEncoder

from backend.core.config import settings
from backend.core.models import ScoredPassage

class CrossEncoderReranker:
    _instance = None

    def __init__(self, model_name: str = settings.RERANKER_MODEL_NAME):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model_name = model_name
        self.model = CrossEncoder(model_name, device=device)

    @classmethod
    def get_instance(cls, model_name: str = settings.RERANKER_MODEL_NAME) -> "CrossEncoderReranker":
        if cls._instance is None or cls._instance.model_name != model_name:
            cls._instance = cls(model_name)
        return cls._instance

    def rerank(self, query: str, passages: List[ScoredPassage], top_k: int = 5) -> List[ScoredPassage]:
        if not passages:
            return []
            
        pairs = [[query, p.text] for p in passages]
        raw_scores = self.model.predict(pairs, show_progress_bar=False)
        
        # Apply sigmoid to convert logits to [0, 1] if not already bounded
        if isinstance(raw_scores, np.ndarray):
            scores_list = raw_scores.tolist()
        else:
            scores_list = list(raw_scores)
            
        # Sigmoid function for stable normalization
        def sigmoid(x):
            return 1.0 / (1.0 + np.exp(-x))

        scored_passages: List[ScoredPassage] = []
        for p, raw_s in zip(passages, scores_list):
            prob_score = float(sigmoid(raw_s)) if (raw_s < 0 or raw_s > 1) else float(raw_s)
            p_copy = p.model_copy(update={"rerank_score": prob_score})
            scored_passages.append(p_copy)
            
        # Sort descending by rerank_score
        scored_passages.sort(key=lambda x: x.rerank_score or 0.0, reverse=True)
        return scored_passages[:top_k]
