"""
EvidenceGuard Core Data Models
Pydantic v2 schemas for all system entities, graph state payloads, and API contracts.
"""

from enum import Enum
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

class EpistemicDecision(str, Enum):
    SUFFICIENT = "SUFFICIENT"
    PARTIALLY_SUFFICIENT = "PARTIALLY_SUFFICIENT"
    INSUFFICIENT = "INSUFFICIENT"
    CONFLICTING = "CONFLICTING"
    IRRELEVANT = "IRRELEVANT"

class DocumentMetadata(BaseModel):
    doc_id: str
    title: str
    source: str
    file_type: str
    date: Optional[str] = None
    author: Optional[str] = None
    version: int = 1
    content_hash: Optional[str] = None
    extra: Dict[str, Any] = Field(default_factory=dict)

class Document(BaseModel):
    id: str
    text: str
    metadata: DocumentMetadata

class Chunk(BaseModel):
    id: str
    doc_id: str
    text: str
    chunk_index: int
    parent_id: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    embedding: Optional[List[float]] = None

class ScoredPassage(BaseModel):
    id: str
    doc_id: str
    title: str
    text: str
    dense_score: Optional[float] = None
    bm25_score: Optional[float] = None
    rrf_score: Optional[float] = None
    rerank_score: Optional[float] = None
    is_gold: Optional[bool] = None
    date: Optional[str] = None
    source: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

class ClaimVerification(BaseModel):
    claim: str
    supported: bool
    citation_id: Optional[str] = None
    entailment_score: float
    neutral_score: float
    contradiction_score: float
    reasoning: Optional[str] = None

class Citation(BaseModel):
    citation_id: str
    chunk_id: str
    source_title: str
    snippet: str
    supported_claims: List[str] = Field(default_factory=list)
    verified: bool = True

class ConflictReport(BaseModel):
    has_conflict: bool
    conflicting_claims: List[str] = Field(default_factory=list)
    source_passage_ids: List[str] = Field(default_factory=list)
    explanation: Optional[str] = None
    is_temporal: bool = False
    resolved: bool = False
    preferred_claim: Optional[str] = None

class EvidenceAssessment(BaseModel):
    decision: EpistemicDecision
    confidence: float
    predicted_probabilities: Dict[str, float] = Field(default_factory=dict)
    features_used: Dict[str, float] = Field(default_factory=dict)
    explanation: str

class QueryRequest(BaseModel):
    query: str
    mode: str = Field(default="evidenceguard", description="'baseline' or 'evidenceguard'")
    top_k: int = 5
    llm_provider: Optional[str] = None
    temperature: float = 0.0
    filters: Optional[Dict[str, Any]] = None

class QueryResponse(BaseModel):
    query: str
    mode: str
    answer: str
    decision: EpistemicDecision
    confidence: float
    citations: List[Citation] = Field(default_factory=list)
    retrieved_passages: List[ScoredPassage] = Field(default_factory=list)
    conflicts: Optional[ConflictReport] = None
    claim_verifications: List[ClaimVerification] = Field(default_factory=list)
    abstention_reason: Optional[str] = None
    execution_trace: List[Dict[str, Any]] = Field(default_factory=list)
    latency_ms: float = 0.0
