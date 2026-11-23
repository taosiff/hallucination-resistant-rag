"""
EvidenceGuard BM25 Sparse Lexical Retriever
Token-level Okapi BM25 retrieval with score normalization.
"""

import re
from typing import List, Dict, Any, Optional
from rank_bm25 import BM25Okapi

from backend.core.models import Chunk, ScoredPassage

def tokenize_corpus_text(text: str) -> List[str]:
    # Lowercase and split on non-alphanumeric characters
    tokens = re.findall(r"\b[a-zA-Z0-9_\-\.]+\b", text.lower())
    return [t for t in tokens if len(t) > 1]

class BM25Retriever:
    def __init__(self):
        self.corpus_chunks: List[Chunk] = []
        self.tokenized_corpus: List[List[str]] = []
        self.bm25: Optional[BM25Okapi] = None

    def index_chunks(self, chunks: List[Chunk]):
        self.corpus_chunks = chunks
        self.tokenized_corpus = [tokenize_corpus_text(c.text) for c in chunks]
        if self.tokenized_corpus:
            self.bm25 = BM25Okapi(self.tokenized_corpus)
        else:
            self.bm25 = None

    def search(self, query: str, top_k: int = 10) -> List[ScoredPassage]:
        if not self.bm25 or not self.corpus_chunks:
            return []
            
        tokenized_query = tokenize_corpus_text(query)
        if not tokenized_query:
            return []
            
        raw_scores = self.bm25.get_scores(tokenized_query)
        top_indices = sorted(range(len(raw_scores)), key=lambda i: raw_scores[i], reverse=True)[:top_k]
        
        max_score = max(raw_scores) if len(raw_scores) > 0 and max(raw_scores) > 0 else 1.0
        
        passages: List[ScoredPassage] = []
        for idx in top_indices:
            score = float(raw_scores[idx])
            if score <= 0:
                continue
            chunk = self.corpus_chunks[idx]
            norm_score = min(1.0, score / max_score)
            passages.append(ScoredPassage(
                id=chunk.id,
                doc_id=chunk.doc_id,
                title=chunk.metadata.get("title", chunk.doc_id),
                text=chunk.text,
                bm25_score=norm_score,
                date=chunk.metadata.get("date"),
                source=chunk.metadata.get("source"),
                metadata=chunk.metadata
            ))
        return passages
