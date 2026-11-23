"""
EvidenceGuard LangGraph State Definition
Strictly typed Pydantic state schema for LangGraph agentic workflow execution.
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

from backend.core.models import (
    ScoredPassage,
    Citation,
    ClaimVerification,
    ConflictReport,
    EvidenceAssessment,
    EpistemicDecision
)

class GraphState(BaseModel):
    # Query Information
    original_query: str
    current_query: str
    rewritten_queries: List[str] = Field(default_factory=list)
    query_intent: Optional[str] = None
    entities: List[str] = Field(default_factory=list)
    
    # Retrieval & Reranking
    retrieved_documents: List[ScoredPassage] = Field(default_factory=list)
    reranked_documents: List[ScoredPassage] = Field(default_factory=list)
    retrieval_metadata: Dict[str, Any] = Field(default_factory=dict)
    
    # Evidence Assessment & Epistemics
    evidence_assessment: Optional[EvidenceAssessment] = None
    epistemic_decision: EpistemicDecision = EpistemicDecision.INSUFFICIENT
    confidence_score: float = 0.0
    
    # Conflicts
    conflict_report: Optional[ConflictReport] = None
    
    # Generation & Verification
    raw_answer: Optional[str] = None
    final_answer: Optional[str] = None
    supporting_claims: List[ClaimVerification] = Field(default_factory=list)
    citations: List[Citation] = Field(default_factory=list)
    abstention_reason: Optional[str] = None
    
    # Graph Execution Control
    retry_count: int = 0
    max_retries: int = 2
    execution_trace: List[Dict[str, Any]] = Field(default_factory=list)
    node_timings: Dict[str, float] = Field(default_factory=dict)
    llm_provider: Optional[str] = None
    temperature: float = 0.0
