"""
EvidenceGuard Verification & API Tests
Tests atomic claim decomposition, citation validation, conflict detection, and FastAPI endpoints.
"""

import pytest
from fastapi.testclient import TestClient

from backend.core.models import ScoredPassage
from backend.verification.conflict_detector import ConflictDetector
from backend.verification.claim_verifier import ClaimVerifier
from backend.api.main import app

client = TestClient(app)

def test_conflict_detector_identifies_contradiction():
    detector = ConflictDetector()
    p1 = ScoredPassage(id="p1", doc_id="d1", title="Report A", text="Boeing delivered 396 aircraft in 2023.", date="2023-01-01")
    p2 = ScoredPassage(id="p2", doc_id="d2", title="Report B", text="Boeing delivered 440 aircraft in 2023 however analysts differed.", date="2023-02-01")
    
    report = detector.detect_conflicts([p1, p2])
    assert report.has_conflict is True
    assert len(report.conflicting_claims) == 2

def test_claim_verifier_catches_unsupported_claim():
    verifier = ClaimVerifier()
    passages = [
        ScoredPassage(id="p1", doc_id="d1", title="Biology", text="Photosynthesis takes place in the chloroplasts of plant cells.")
    ]
    # Answer contains one grounded sentence and one complete fabrication
    answer = "Photosynthesis takes place in chloroplasts [p1]. Also, human beings have three hearts and breathe argon gas."
    verifications, citations = verifier.verify_claims(answer, passages)
    
    assert len(verifications) == 2
    assert verifications[0].supported is True
    assert verifications[1].supported is False
    assert verifications[1].citation_id is None

def test_api_health():
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert "device" in data

def test_api_query_baseline_and_evidenceguard():
    res_eg = client.post("/query", json={"query": "What is metformin?", "mode": "evidenceguard"})
    assert res_eg.status_code == 200
    data_eg = res_eg.json()
    assert "decision" in data_eg
    assert "latency_ms" in data_eg

    res_base = client.post("/query", json={"query": "What is metformin?", "mode": "baseline"})
    assert res_base.status_code == 200
    data_base = res_base.json()
    assert data_base["mode"] == "baseline"

def test_api_experiments_listing():
    res = client.get("/experiments")
    assert res.status_code == 200
    data = res.json()
    assert len(data["experiments"]) >= 4
