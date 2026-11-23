"""
EvidenceGuard Embeddings Interface
Sentence-Transformers dense embedding wrapper with CUDA acceleration and normalization.
"""

from typing import List
import torch
from sentence_transformers import SentenceTransformer

from backend.core.config import settings

class EmbeddingService:
    _instance = None
    
    def __init__(self, model_name: str = settings.EMBEDDING_MODEL_NAME):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device=device)
        self.dimension = self.model.get_embedding_dimension()
        
    @classmethod
    def get_instance(cls, model_name: str = settings.EMBEDDING_MODEL_NAME) -> "EmbeddingService":
        if cls._instance is None or cls._instance.model_name != model_name:
            cls._instance = cls(model_name)
        return cls._instance

    def embed_query(self, query: str) -> List[float]:
        vec = self.model.encode(query, normalize_embeddings=True, show_progress_bar=False)
        return vec.tolist()

    def embed_documents(self, texts: List[str], batch_size: int = 64) -> List[List[float]]:
        if not texts:
            return []
        vecs = self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=False
        )
        return vecs.tolist()
