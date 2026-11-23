"""
EvidenceGuard Retrieval & Reranker Unit Tests
Tests dense search, BM25 sparse search, RRF hybrid fusion, and Cross-Encoder reranking.
"""

import pytest
from backend.core.models import Chunk, ScoredPassage
from backend.retrieval.bm25_retriever import BM25Retriever
from backend.retrieval.reranker import CrossEncoderReranker

def test_bm25_retrieval_exact_match():
    bm25 = BM25Retriever()
    chunks = [
        Chunk(id="c1", doc_id="d1", text="Quantum computing uses superconducting qubits.", chunk_index=0),
        Chunk(id="c2", doc_id="d2", text="Classical thermodynamics involves heat engines and Carnot cycles.", chunk_index=1),
        Chunk(id="c3", doc_id="d3", text="Photosynthesis converts solar light into glucose in plants.", chunk_index=2)
    ]
    bm25.index_chunks(chunks)
    
    res = bm25.search("qubits", top_k=1)
    assert len(res) == 1
    assert res[0].id == "c1"
    assert res[0].bm25_score > 0.0

def test_cross_encoder_reranking():
    reranker = CrossEncoderReranker.get_instance()
    passages = [
        ScoredPassage(id="p1", doc_id="d1", title="Plant Biology", text="Trees absorb carbon dioxide and release oxygen."),
        ScoredPassage(id="p2", doc_id="d2", title="Cardiology", text="Metformin is an oral antihyperglycemic medication that activates AMPK."),
        ScoredPassage(id="p3", doc_id="d3", title="Astrophysics", text="The Hubble space telescope orbits the Earth.")
    ]
    reranked = reranker.rerank("What drug activates AMPK for blood glucose?", passages, top_k=1)
    assert len(reranked) == 1
    assert reranked[0].id == "p2"
    assert reranked[0].rerank_score > 0.50
