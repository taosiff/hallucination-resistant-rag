"""
EvidenceGuard Hybrid Retrieval System
Fuses dense semantic search and BM25 sparse lexical search via Reciprocal Rank Fusion (RRF)
and optional Cross-Encoder reranking.
"""

from typing import List, Dict, Any, Optional
from collections import defaultdict

from backend.core.config import settings
from backend.core.models import Chunk, ScoredPassage
from backend.retrieval.vector_store import ChromaVectorStore
from backend.retrieval.bm25_retriever import BM25Retriever
from backend.retrieval.reranker import CrossEncoderReranker

class HybridRetriever:
    def __init__(
        self,
        vector_store: Optional[ChromaVectorStore] = None,
        bm25_retriever: Optional[BM25Retriever] = None,
        reranker: Optional[CrossEncoderReranker] = None
    ):
        self.vector_store = vector_store or ChromaVectorStore()
        self.bm25_retriever = bm25_retriever or BM25Retriever()
        self.reranker = reranker or CrossEncoderReranker.get_instance()

    def index_chunks(self, chunks: List[Chunk]):
        self.vector_store.add_chunks(chunks)
        self.bm25_retriever.index_chunks(chunks)

    def retrieve_dense(self, query: str, top_k: int = 15, filters: Optional[Dict[str, Any]] = None) -> List[ScoredPassage]:
        return self.vector_store.search(query, top_k=top_k, where_filter=filters)

    def retrieve_sparse(self, query: str, top_k: int = 15) -> List[ScoredPassage]:
        return self.bm25_retriever.search(query, top_k=top_k)

    def retrieve_hybrid_rrf(
        self,
        query: str,
        dense_top_k: int = 15,
        sparse_top_k: int = 15,
        rrf_k: int = 60,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[ScoredPassage]:
        dense_results = self.retrieve_dense(query, top_k=dense_top_k, filters=filters)
        sparse_results = self.retrieve_sparse(query, top_k=sparse_top_k)
        
        # Merge via RRF
        rrf_scores: Dict[str, float] = defaultdict(float)
        passage_map: Dict[str, ScoredPassage] = {}
        
        for rank, p in enumerate(dense_results):
            rrf_scores[p.id] += 1.0 / (rrf_k + rank + 1)
            passage_map[p.id] = p
            
        for rank, p in enumerate(sparse_results):
            rrf_scores[p.id] += 1.0 / (rrf_k + rank + 1)
            if p.id not in passage_map:
                passage_map[p.id] = p
            else:
                # Merge scores
                passage_map[p.id].bm25_score = p.bm25_score
                
        # Build unified scored list
        unified: List[ScoredPassage] = []
        for p_id, rrf_s in rrf_scores.items():
            p = passage_map[p_id].model_copy(update={"rrf_score": rrf_s})
            unified.append(p)
            
        unified.sort(key=lambda x: x.rrf_score or 0.0, reverse=True)
        return unified

    def retrieve(
        self,
        query: str,
        mode: str = "hybrid_rerank",
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[ScoredPassage]:
        """
        Unified retrieval entrypoint supporting:
        - 'dense'
        - 'sparse'
        - 'hybrid' (RRF only)
        - 'hybrid_rerank' (Hybrid RRF + Cross-Encoder reranking)
        """
        if mode == "dense":
            return self.retrieve_dense(query, top_k=top_k, filters=filters)
        elif mode == "sparse":
            return self.retrieve_sparse(query, top_k=top_k)
        elif mode == "hybrid":
            return self.retrieve_hybrid_rrf(query, dense_top_k=top_k*3, sparse_top_k=top_k*3, filters=filters)[:top_k]
        else: # default hybrid_rerank
            candidates = self.retrieve_hybrid_rrf(
                query,
                dense_top_k=settings.DENSE_TOP_K,
                sparse_top_k=settings.SPARSE_TOP_K,
                filters=filters
            )
            # Rerank top candidates with cross-encoder
            reranked = self.reranker.rerank(query, candidates, top_k=top_k)
            return reranked
