"""
EvidenceGuard LangGraph Agentic Workflow
Coordinates query analysis, hybrid retrieval, cross-encoder reranking,
epistemic evidence assessment, conflict detection, claim verification, and calibrated abstention.
"""

import time
from typing import Dict, Any, Optional
from langgraph.graph import StateGraph, END

from backend.core.config import settings
from backend.core.models import (
    QueryRequest,
    QueryResponse,
    EpistemicDecision
)
from backend.graph.state import GraphState
from backend.graph.nodes import (
    analyze_query,
    rewrite_query,
    retrieve_documents,
    rerank_documents,
    assess_evidence,
    detect_conflicts,
    decide_answerability,
    generate_answer,
    verify_claims,
    validate_citations,
    abstain,
    finalize_response
)

def route_decision(state: GraphState) -> str:
    """Conditional router dispatching workflow based on epistemic assessment."""
    decision = state.epistemic_decision
    
    if decision == EpistemicDecision.SUFFICIENT:
        return "generate_answer"
    elif decision == EpistemicDecision.PARTIALLY_SUFFICIENT:
        return "generate_answer"
    elif decision == EpistemicDecision.CONFLICTING:
        return "detect_conflicts"
    elif decision == EpistemicDecision.IRRELEVANT:
        if state.retry_count < state.max_retries:
            return "rewrite_query"
        return "abstain"
    else:  # INSUFFICIENT
        return "abstain"

def build_evidenceguard_graph():
    builder = StateGraph(GraphState)
    
    # 1. Add all 12 nodes
    builder.add_node("analyze_query", analyze_query)
    builder.add_node("rewrite_query", rewrite_query)
    builder.add_node("retrieve_documents", retrieve_documents)
    builder.add_node("rerank_documents", rerank_documents)
    builder.add_node("assess_evidence", assess_evidence)
    builder.add_node("detect_conflicts", detect_conflicts)
    builder.add_node("decide_answerability", decide_answerability)
    builder.add_node("generate_answer", generate_answer)
    builder.add_node("verify_claims", verify_claims)
    builder.add_node("validate_citations", validate_citations)
    builder.add_node("abstain", abstain)
    builder.add_node("finalize_response", finalize_response)
    
    # 2. Add linear pre-assessment edges
    builder.set_entry_point("analyze_query")
    builder.add_edge("analyze_query", "retrieve_documents")
    builder.add_edge("retrieve_documents", "rerank_documents")
    builder.add_edge("rerank_documents", "assess_evidence")
    builder.add_edge("assess_evidence", "decide_answerability")
    
    # 3. Add conditional routing edges
    builder.add_conditional_edges(
        "decide_answerability",
        route_decision,
        {
            "generate_answer": "generate_answer",
            "detect_conflicts": "detect_conflicts",
            "rewrite_query": "rewrite_query",
            "abstain": "abstain"
        }
    )
    
    # 4. Conflict path
    builder.add_edge("detect_conflicts", "generate_answer")
    
    # 5. Retry loop
    builder.add_edge("rewrite_query", "retrieve_documents")
    
    # 6. Verification and finalization edges
    builder.add_edge("generate_answer", "verify_claims")
    builder.add_edge("verify_claims", "validate_citations")
    builder.add_edge("validate_citations", "finalize_response")
    builder.add_edge("abstain", "finalize_response")
    builder.add_edge("finalize_response", END)
    
    return builder.compile()

class EvidenceGuardRAG:
    _compiled_graph = None

    def __init__(self):
        if EvidenceGuardRAG._compiled_graph is None:
            EvidenceGuardRAG._compiled_graph = build_evidenceguard_graph()
        self.graph = EvidenceGuardRAG._compiled_graph

    def query(self, request: QueryRequest) -> QueryResponse:
        t0 = time.perf_counter()
        
        initial_state = GraphState(
            original_query=request.query,
            current_query=request.query,
            llm_provider=request.llm_provider,
            temperature=request.temperature
        )
        
        result_dict = self.graph.invoke(initial_state)
        # LangGraph returns dictionary representing the state
        state = GraphState.model_validate(result_dict)
        latency = (time.perf_counter() - t0) * 1000.0
        
        return QueryResponse(
            query=request.query,
            mode="evidenceguard",
            answer=state.final_answer or state.raw_answer or "No response generated.",
            decision=state.epistemic_decision,
            confidence=state.confidence_score,
            citations=state.citations,
            retrieved_passages=state.reranked_documents or state.retrieved_documents,
            conflicts=state.conflict_report,
            claim_verifications=state.supporting_claims,
            abstention_reason=state.abstention_reason,
            execution_trace=state.execution_trace,
            latency_ms=latency
        )
