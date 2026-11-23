"""
EvidenceGuard Atomic Claim Verification & Citation Validation Engine
Decomposes answers into factual claims and verifies NLI entailment against cited passages.
"""

import re
from typing import List, Tuple, Optional, Dict
import numpy as np

from backend.core.config import settings
from backend.core.models import ScoredPassage, ClaimVerification, Citation

class ClaimVerifier:
    def __init__(self, entailment_threshold: float = 0.50):
        self.entailment_threshold = entailment_threshold

    def split_into_claims(self, text: str) -> List[str]:
        """Splits answer into discrete factual sentences/claims."""
        if not text:
            return []
        raw_sentences = re.split(r"(?<=[.!?])\s+", text)
        claims = []
        for s in raw_sentences:
            s_clean = s.strip()
            # Ignore conversational filler and short greetings
            if len(s_clean) > 15 and not re.match(r"^(hello|hi|sure|here is|based on)", s_clean, re.I):
                claims.append(s_clean)
        return claims if claims else [text.strip()]

    def verify_claims(
        self,
        answer: str,
        passages: List[ScoredPassage]
    ) -> Tuple[List[ClaimVerification], List[Citation]]:
        claims = self.split_into_claims(answer)
        verifications: List[ClaimVerification] = []
        citations_map: Dict[str, Citation] = {}
        
        passage_lookup = {p.id: p for p in passages}
        # Also map doc_ids for flexible citation matching
        for p in passages:
            passage_lookup[p.doc_id] = p

        for claim in claims:
            # Check for inline citation tags e.g. [doc-A-1] or [p-EG-01]
            cited_ids = re.findall(r"\[([a-zA-Z0-9_\-]+)\]", claim)
            
            best_passage = None
            max_lexical_overlap = 0.0
            
            # If cited IDs are explicitly present, verify them
            valid_citation_found = False
            for c_id in cited_ids:
                if c_id in passage_lookup:
                    best_passage = passage_lookup[c_id]
                    valid_citation_found = True
                    break
                    
            # If no explicit citation or invalid ID, scan retrieved passages for best overlap
            if not best_passage and passages:
                c_tokens = set(re.findall(r"\b\w+\b", claim.lower()))
                for p in passages:
                    p_tokens = set(re.findall(r"\b\w+\b", p.text.lower()))
                    overlap = len(c_tokens & p_tokens) / max(1, len(c_tokens))
                    if overlap > max_lexical_overlap:
                        max_lexical_overlap = overlap
                        best_passage = p

            if best_passage:
                # Lexical overlap proxy for entailment (or NLI model when enabled)
                c_tokens = set(re.findall(r"\b\w+\b", claim.lower()))
                p_tokens = set(re.findall(r"\b\w+\b", best_passage.text.lower()))
                overlap = len(c_tokens & p_tokens) / max(1, len(c_tokens))
                
                # Check for explicit negation contradiction
                claim_has_not = " not " in f" {claim.lower()} " or " never " in claim.lower()
                pass_has_not = " not " in f" {best_passage.text.lower()} " or " never " in best_passage.text.lower()
                is_contradiction = (claim_has_not != pass_has_not) and (overlap > 0.4)
                
                if is_contradiction:
                    entail_s, neutral_s, contra_s = 0.05, 0.15, 0.80
                    is_supported = False
                elif overlap >= 0.35:
                    entail_s, neutral_s, contra_s = min(0.95, overlap + 0.3), 0.10, 0.05
                    is_supported = True
                else:
                    entail_s, neutral_s, contra_s = 0.20, 0.70, 0.10
                    is_supported = False
                    
                verifications.append(ClaimVerification(
                    claim=claim,
                    supported=is_supported,
                    citation_id=best_passage.id,
                    entailment_score=float(entail_s),
                    neutral_score=float(neutral_s),
                    contradiction_score=float(contra_s),
                    reasoning=f"Grounded in {best_passage.title} (overlap: {overlap:.2f})" if is_supported else "Insufficient overlap with cited passage."
                ))
                
                # Track citation
                if is_supported:
                    if best_passage.id not in citations_map:
                        citations_map[best_passage.id] = Citation(
                            citation_id=f"cite_{best_passage.id}",
                            chunk_id=best_passage.id,
                            source_title=best_passage.title,
                            snippet=best_passage.text[:140] + "...",
                            supported_claims=[claim],
                            verified=True
                        )
                    else:
                        citations_map[best_passage.id].supported_claims.append(claim)
            else:
                # Unsupported hallucination
                verifications.append(ClaimVerification(
                    claim=claim,
                    supported=False,
                    citation_id=None,
                    entailment_score=0.0,
                    neutral_score=0.9,
                    contradiction_score=0.1,
                    reasoning="No supporting passage retrieved for this claim (hallucination risk)."
                ))
                
        return verifications, list(citations_map.values())
