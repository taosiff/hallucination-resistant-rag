"""
EvidenceGuard Evidence Sufficiency & Epistemic Classifier Unit Tests
Tests 5-way decision routing, feature extraction, calibration, and edge cases.
"""

import pytest
from backend.core.models import ScoredPassage, EpistemicDecision
from backend.evidence.features import extract_evidence_features
from backend.evidence.classifier import EvidenceSufficiencyClassifier

def test_feature_extraction_empty_passages():
    feats = extract_evidence_features("What is X?", [])
    assert feats["top_rerank_score"] == 0.0
    assert feats["num_passages_above_threshold"] == 0.0

def test_classifier_sufficient_evidence():
    classifier = EvidenceSufficiencyClassifier()
    passages = [
        ScoredPassage(id="p1", doc_id="d1", title="Metformin", text="Metformin reduces hepatic gluconeogenesis via AMPK.", dense_score=0.85, rerank_score=0.92, bm25_score=0.88),
        ScoredPassage(id="p2", doc_id="d2", title="Diabetes", text="Insulin regulates blood sugar in humans.", dense_score=0.40, rerank_score=0.20, bm25_score=0.15)
    ]
    assessment = classifier.predict("How does metformin reduce hepatic glucose?", passages)
    assert assessment.decision in [EpistemicDecision.SUFFICIENT, EpistemicDecision.PARTIALLY_SUFFICIENT]
    assert assessment.confidence > 0.60

def test_classifier_irrelevant_distractors():
    classifier = EvidenceSufficiencyClassifier()
    passages = [
        ScoredPassage(id="p1", doc_id="d1", title="Everest", text="Mount Everest is Earth's highest peak in the Himalayas.", dense_score=0.65, rerank_score=0.15, bm25_score=0.05)
    ]
    assessment = classifier.predict("What is the acute mountain sickness dosage?", passages)
    assert assessment.decision in [EpistemicDecision.IRRELEVANT, EpistemicDecision.INSUFFICIENT]
