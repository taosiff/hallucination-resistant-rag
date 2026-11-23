"""
EvidenceGuard Dedicated Conflict Detection Engine
Identifies factual disagreements, numerical discrepancies, and temporal obsolescence
across retrieved passages.
"""

import re
from typing import List, Optional
from backend.core.models import ScoredPassage, ConflictReport

class ConflictDetector:
    def __init__(self, contradiction_threshold: float = 0.40):
        self.contradiction_threshold = contradiction_threshold

    def detect_conflicts(self, passages: List[ScoredPassage]) -> ConflictReport:
        if len(passages) < 2:
            return ConflictReport(has_conflict=False)

        p1 = passages[0]
        p2 = passages[1]
        t1 = p1.text.lower()
        t2 = p2.text.lower()

        # Check for temporal contradiction if dates exist
        is_temporal = False
        preferred_claim = None
        if p1.date and p2.date and p1.date != p2.date:
            is_temporal = True
            newer_p = p1 if p1.date > p2.date else p2
            older_p = p2 if p1.date > p2.date else p1
            preferred_claim = f"Source ({newer_p.title}, {newer_p.date}) supersedes older source ({older_p.title}, {older_p.date})."

        # Check for numeric discrepancies
        nums1 = set(re.findall(r"\b\d{4}\b|\b\d{2,}\b", t1))
        nums2 = set(re.findall(r"\b\d{4}\b|\b\d{2,}\b", t2))
        
        conflict_tokens = [
            "however", "dispute", "instead", "whereas", "contradicts", 
            "demoted", "retired", "differed", "replaced", "earlier", "previously"
        ]
        has_conflict_cue = any(c in t1 or c in t2 for c in conflict_tokens)

        # Explicit conflict conditions
        if (nums1 and nums2 and nums1 != nums2 and has_conflict_cue) or is_temporal:
            claims = [
                f"{p1.title} ({p1.date or 'N/A'}): {p1.text[:120]}...",
                f"{p2.title} ({p2.date or 'N/A'}): {p2.text[:120]}..."
            ]
            explanation = "Temporal change or factual discrepancy detected between sources." if is_temporal else "Direct factual discrepancy detected between retrieved documents."
            return ConflictReport(
                has_conflict=True,
                conflicting_claims=claims,
                source_passage_ids=[p1.id, p2.id],
                explanation=explanation,
                is_temporal=is_temporal,
                resolved=is_temporal,
                preferred_claim=preferred_claim
            )

        return ConflictReport(has_conflict=False)
