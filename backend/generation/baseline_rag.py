"""
EvidenceGuard Baseline RAG Implementation
Standard Naive RAG pipeline: Question -> Dense Retrieve -> Construct Context -> Generate Answer -> Return Sources.
Used as the reference baseline to quantify improvements from EvidenceGuard.
"""

import time
from typing import Optional, Dict, Any, List

from backend.core.config import settings
from backend.core.models import (
    QueryRequest,
    QueryResponse,
    ScoredPassage,
    Citation,
    EpistemicDecision
)
from backend.retrieval.hybrid_retriever import HybridRetriever
from backend.generation.providers import LLMProviderFactory, BaseLLMProvider

BASELINE_SYSTEM_PROMPT = """You are a helpful assistant. Answer the user's question using ONLY the provided context passages.
- Cite the supporting document IDs in square brackets (e.g. [doc_id]).
- If the context does not contain the answer, say "I don't know based on the provided context."
- Do not make up facts or extrapolate beyond what is stated."""

class BaselineRAG:
    def __init__(self, retriever: Optional[HybridRetriever] = None, llm_provider: Optional[BaseLLMProvider] = None):
        self.retriever = retriever or HybridRetriever()
        self.llm = llm_provider or LLMProviderFactory.get_provider()

    def query(self, request: QueryRequest) -> QueryResponse:
        start_time = time.perf_counter()
        
        # 1. Dense retrieval only (Standard Naive RAG)
        passages = self.retriever.retrieve_dense(
            query=request.query,
            top_k=request.top_k,
            filters=request.filters
        )
        
        # 2. Build context
        context_blocks = []
        for p in passages:
            context_blocks.append(f"[{p.id}] (Title: {p.title}):\n{p.text}")
        context_str = "\n\n".join(context_blocks)
        
        # 3. Build prompt
        prompt = f"""Context:
{context_str}

Question: {request.query}

Answer:"""

        # 4. Generate answer
        llm = LLMProviderFactory.get_provider(request.llm_provider) if request.llm_provider else self.llm
        raw_answer = llm.generate(
            prompt=prompt,
            system_prompt=BASELINE_SYSTEM_PROMPT,
            temperature=request.temperature
        )
        
        # 5. Extract naive citations
        citations = []
        for p in passages:
            if p.id in raw_answer or p.doc_id in raw_answer:
                citations.append(Citation(
                    citation_id=f"cite_{p.id}",
                    chunk_id=p.id,
                    source_title=p.title,
                    snippet=p.text[:150] + "...",
                    verified=False  # Baseline does not verify
                ))
                
        latency = (time.perf_counter() - start_time) * 1000.0
        
        # Baseline always declares SUFFICIENT if non-empty answer, else INSUFFICIENT
        decision = EpistemicDecision.INSUFFICIENT if ("don't know" in raw_answer.lower() or not raw_answer.strip()) else EpistemicDecision.SUFFICIENT
        
        return QueryResponse(
            query=request.query,
            mode="baseline",
            answer=raw_answer,
            decision=decision,
            confidence=0.5,  # Baseline has no calibrated confidence
            citations=citations,
            retrieved_passages=passages,
            conflicts=None,
            claim_verifications=[],
            abstention_reason="I don't know based on the provided context." if decision == EpistemicDecision.INSUFFICIENT else None,
            execution_trace=[
                {"node": "dense_retrieve", "passages_found": len(passages)},
                {"node": "generate_answer", "model": getattr(llm, "model", "mock")}
            ],
            latency_ms=latency
        )
