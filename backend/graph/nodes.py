"""
EvidenceGuard LangGraph Node Implementations
Contains the 12 specialized graph nodes executing the evidence-first RAG pipeline.
"""

import time
import re
from typing import Dict, Any, List

from backend.core.config import settings
from backend.core.models import (
    ScoredPassage,
    EpistemicDecision,
    Citation,
    ClaimVerification,
    ConflictReport,
    EvidenceAssessment
)
from backend.graph.state import GraphState
from backend.retrieval.hybrid_retriever import HybridRetriever
from backend.retrieval.reranker import CrossEncoderReranker
from backend.evidence.classifier import EvidenceSufficiencyClassifier
from backend.verification.conflict_detector import ConflictDetector
from backend.verification.claim_verifier import ClaimVerifier
from backend.generation.providers import LLMProviderFactory

# Singletons for graph execution
retriever_instance = HybridRetriever()
reranker_instance = CrossEncoderReranker.get_instance()
classifier_instance = EvidenceSufficiencyClassifier()
conflict_detector = ConflictDetector()
claim_verifier = ClaimVerifier()

def analyze_query(state: GraphState) -> Dict[str, Any]:
    t0 = time.perf_counter()
    q = state.current_query
    
    # Extract entities and key concepts
    tokens = re.findall(r"\b[A-Za-z0-9\-_]{3,}\b", q)
    # Detect intent
    intent = "factoid"
    if any(w in q.lower() for w in ["why", "how", "explain", "mechanism"]):
        intent = "explanatory"
    elif any(w in q.lower() for w in ["and", "both", "where was", "who founded"]):
        intent = "multi_part"
        
    trace_entry = {
        "node": "analyze_query",
        "query": q,
        "intent": intent,
        "tokens_extracted": len(tokens)
    }
    return {
        "query_intent": intent,
        "entities": tokens[:5],
        "execution_trace": state.execution_trace + [trace_entry],
        "node_timings": {**state.node_timings, "analyze_query": (time.perf_counter() - t0) * 1000}
    }

def rewrite_query(state: GraphState) -> Dict[str, Any]:
    t0 = time.perf_counter()
    new_retry = state.retry_count + 1
    orig = state.original_query
    
    # Query reformulation heuristics: strip conversational fluff, expand key entities
    cleaned = re.sub(r"(?i)\b(please|can you tell me|what is|how do i find|explain to me)\b", "", orig).strip()
    rewritten = f"{cleaned} overview details"
    
    trace_entry = {
        "node": "rewrite_query",
        "retry_count": new_retry,
        "rewritten_query": rewritten
    }
    return {
        "current_query": rewritten,
        "rewritten_queries": state.rewritten_queries + [rewritten],
        "retry_count": new_retry,
        "execution_trace": state.execution_trace + [trace_entry],
        "node_timings": {**state.node_timings, "rewrite_query": (time.perf_counter() - t0) * 1000}
    }

def retrieve_documents(state: GraphState) -> Dict[str, Any]:
    t0 = time.perf_counter()
    query = state.current_query
    # Perform hybrid dense + sparse retrieval
    candidates = retriever_instance.retrieve_hybrid_rrf(
        query=query,
        dense_top_k=settings.DENSE_TOP_K,
        sparse_top_k=settings.SPARSE_TOP_K
    )
    trace_entry = {
        "node": "retrieve_documents",
        "query": query,
        "candidates_retrieved": len(candidates)
    }
    return {
        "retrieved_documents": candidates,
        "execution_trace": state.execution_trace + [trace_entry],
        "node_timings": {**state.node_timings, "retrieve_documents": (time.perf_counter() - t0) * 1000}
    }

def rerank_documents(state: GraphState) -> Dict[str, Any]:
    t0 = time.perf_counter()
    candidates = state.retrieved_documents
    query = state.current_query
    
    reranked = reranker_instance.rerank(query, candidates, top_k=settings.RERANK_TOP_K)
    trace_entry = {
        "node": "rerank_documents",
        "top_rerank_score": reranked[0].rerank_score if reranked else 0.0,
        "reranked_count": len(reranked)
    }
    return {
        "reranked_documents": reranked,
        "execution_trace": state.execution_trace + [trace_entry],
        "node_timings": {**state.node_timings, "rerank_documents": (time.perf_counter() - t0) * 1000}
    }

def assess_evidence(state: GraphState) -> Dict[str, Any]:
    t0 = time.perf_counter()
    passages = state.reranked_documents or state.retrieved_documents
    assessment = classifier_instance.predict(state.original_query, passages)
    
    trace_entry = {
        "node": "assess_evidence",
        "decision": assessment.decision.value,
        "confidence": assessment.confidence,
        "explanation": assessment.explanation
    }
    return {
        "evidence_assessment": assessment,
        "epistemic_decision": assessment.decision,
        "confidence_score": assessment.confidence,
        "execution_trace": state.execution_trace + [trace_entry],
        "node_timings": {**state.node_timings, "assess_evidence": (time.perf_counter() - t0) * 1000}
    }

def detect_conflicts(state: GraphState) -> Dict[str, Any]:
    t0 = time.perf_counter()
    passages = state.reranked_documents or state.retrieved_documents
    conflict = conflict_detector.detect_conflicts(passages)
    
    trace_entry = {
        "node": "detect_conflicts",
        "has_conflict": conflict.has_conflict,
        "is_temporal": conflict.is_temporal
    }
    return {
        "conflict_report": conflict,
        "execution_trace": state.execution_trace + [trace_entry],
        "node_timings": {**state.node_timings, "detect_conflicts": (time.perf_counter() - t0) * 1000}
    }

def decide_answerability(state: GraphState) -> Dict[str, Any]:
    t0 = time.perf_counter()
    decision = state.epistemic_decision
    
    trace_entry = {
        "node": "decide_answerability",
        "route_selected": decision.value
    }
    return {
        "execution_trace": state.execution_trace + [trace_entry],
        "node_timings": {**state.node_timings, "decide_answerability": (time.perf_counter() - t0) * 1000}
    }

def generate_answer(state: GraphState) -> Dict[str, Any]:
    t0 = time.perf_counter()
    llm = LLMProviderFactory.get_provider(state.llm_provider)
    passages = state.reranked_documents or state.retrieved_documents
    
    # Build context string
    ctx_blocks = []
    for p in passages:
        ctx_blocks.append(f"[{p.id}] (Title: {p.title}, Date: {p.date or 'N/A'}):\n{p.text}")
    ctx_str = "\n\n".join(ctx_blocks)
    
    decision = state.epistemic_decision
    if decision == EpistemicDecision.PARTIALLY_SUFFICIENT:
        system_instruction = (
            "You are EvidenceGuard. The context contains PARTIAL evidence. "
            "Answer ONLY the parts supported by evidence, explicitly state what is missing, and cite passage IDs."
        )
    elif decision == EpistemicDecision.CONFLICTING:
        system_instruction = (
            "You are EvidenceGuard. The context contains CONFLICTING evidence between sources. "
            "Explicitly present both conflicting perspectives, cite their sources, and explain if one is newer."
        )
    else: # SUFFICIENT
        system_instruction = (
            "You are EvidenceGuard. Answer the question using ONLY the provided context passages. "
            "Cite the supporting passage ID for each factual statement (e.g. [p-1]). Do not hallucinate."
        )
        
    prompt = f"Context:\n{ctx_str}\n\nQuestion: {state.original_query}\n\nAnswer:"
    raw_answer = llm.generate(prompt=prompt, system_prompt=system_instruction, temperature=state.temperature)
    
    trace_entry = {
        "node": "generate_answer",
        "decision_context": decision.value,
        "answer_length": len(raw_answer)
    }
    return {
        "raw_answer": raw_answer,
        "final_answer": raw_answer,
        "execution_trace": state.execution_trace + [trace_entry],
        "node_timings": {**state.node_timings, "generate_answer": (time.perf_counter() - t0) * 1000}
    }

def verify_claims(state: GraphState) -> Dict[str, Any]:
    t0 = time.perf_counter()
    answer = state.raw_answer or ""
    passages = state.reranked_documents or state.retrieved_documents
    
    verifications, citations = claim_verifier.verify_claims(answer, passages)
    trace_entry = {
        "node": "verify_claims",
        "claims_extracted": len(verifications),
        "supported_claims": sum(1 for v in verifications if v.supported),
        "verified_citations": len(citations)
    }
    return {
        "supporting_claims": verifications,
        "citations": citations,
        "execution_trace": state.execution_trace + [trace_entry],
        "node_timings": {**state.node_timings, "verify_claims": (time.perf_counter() - t0) * 1000}
    }

def validate_citations(state: GraphState) -> Dict[str, Any]:
    t0 = time.perf_counter()
    # Ensure all citations point to valid retrieved passages
    pass_ids = {p.id for p in state.reranked_documents}
    validated_cites = [c for c in state.citations if c.chunk_id in pass_ids or any(c.chunk_id in p.id for p in state.reranked_documents)]
    
    trace_entry = {
        "node": "validate_citations",
        "valid_citations_retained": len(validated_cites)
    }
    return {
        "citations": validated_cites,
        "execution_trace": state.execution_trace + [trace_entry],
        "node_timings": {**state.node_timings, "validate_citations": (time.perf_counter() - t0) * 1000}
    }

def abstain(state: GraphState) -> Dict[str, Any]:
    t0 = time.perf_counter()
    decision = state.epistemic_decision
    
    if decision == EpistemicDecision.IRRELEVANT:
        reason = "ABSTAIN: Retrieved context passages are semantically irrelevant to the query."
    else:
        reason = "ABSTAIN: The knowledge base does not contain sufficient factual evidence to answer this question."
        
    trace_entry = {
        "node": "abstain",
        "decision": decision.value,
        "abstention_reason": reason
    }
    return {
        "final_answer": reason,
        "abstention_reason": reason,
        "execution_trace": state.execution_trace + [trace_entry],
        "node_timings": {**state.node_timings, "abstain": (time.perf_counter() - t0) * 1000}
    }

def finalize_response(state: GraphState) -> Dict[str, Any]:
    t0 = time.perf_counter()
    trace_entry = {
        "node": "finalize_response",
        "final_decision": state.epistemic_decision.value,
        "total_nodes_executed": len(state.execution_trace) + 1
    }
    return {
        "execution_trace": state.execution_trace + [trace_entry],
        "node_timings": {**state.node_timings, "finalize_response": (time.perf_counter() - t0) * 1000}
    }
