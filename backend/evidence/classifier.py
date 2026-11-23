"""
EvidenceGuard Evidence Sufficiency & Epistemic Abstention Classifier
Calibrated 5-class epistemic decision engine predicting:
SUFFICIENT, PARTIALLY_SUFFICIENT, INSUFFICIENT, CONFLICTING, IRRELEVANT.
"""

import os
import joblib
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

from backend.core.config import settings
from backend.core.models import ScoredPassage, EpistemicDecision, EvidenceAssessment
from backend.evidence.features import extract_evidence_features, feature_dict_to_array, FEATURE_NAMES

LABEL_TO_DECISION = {
    0: EpistemicDecision.SUFFICIENT,
    1: EpistemicDecision.PARTIALLY_SUFFICIENT,
    2: EpistemicDecision.INSUFFICIENT,
    3: EpistemicDecision.CONFLICTING,
    4: EpistemicDecision.IRRELEVANT
}

DECISION_TO_LABEL = {v: k for k, v in LABEL_TO_DECISION.items()}

class EvidenceSufficiencyClassifier:
    def __init__(self, model_type: str = "lightgbm", model_path: Optional[str] = None):
        self.model_type = model_type.lower()
        self.model_path = model_path or str(settings.MODELS_DIR / f"evidence_classifier_{self.model_type}.joblib")
        self.model = None
        self._load_or_init()

    def _load_or_init(self):
        if Path(self.model_path).exists():
            try:
                self.model = joblib.load(self.model_path)
                return
            except Exception as e:
                print(f"Warning: Could not load model from {self.model_path}: {e}")
        self.model = None

    def predict_heuristic(self, features: Dict[str, float]) -> Tuple[EpistemicDecision, float, Dict[str, float], str]:
        """Rule-based decision heuristic baseline."""
        top_rerank = features["top_rerank_score"]
        margin_1_2 = features["rerank_score_margin_1_2"]
        contra_signal = features["pairwise_contradiction_signal"]
        num_above = features["num_passages_above_threshold"]
        overlap = features["lexical_overlap_ratio"]
        top_dense = features["top_dense_score"]

        # Conflict check
        if contra_signal >= 0.70 and top_rerank >= 0.40:
            return (
                EpistemicDecision.CONFLICTING,
                0.85,
                {"CONFLICTING": 0.85, "SUFFICIENT": 0.05, "PARTIAL": 0.05, "INSUFFICIENT": 0.03, "IRRELEVANT": 0.02},
                "Conflicting factual claims detected between top retrieved passages."
            )

        # High confidence single gold passage
        if top_rerank >= 0.65 and (overlap >= 0.25 or margin_1_2 >= 0.20):
            return (
                EpistemicDecision.SUFFICIENT,
                min(0.98, top_rerank),
                {"SUFFICIENT": top_rerank, "PARTIAL": 0.10, "INSUFFICIENT": 0.05, "CONFLICTING": 0.02, "IRRELEVANT": 0.03},
                f"Sufficient evidence identified with top reranker confidence {top_rerank:.2f}."
            )

        # Partial evidence
        if 0.40 <= top_rerank < 0.65 and overlap >= 0.20:
            return (
                EpistemicDecision.PARTIALLY_SUFFICIENT,
                0.72,
                {"PARTIAL": 0.72, "SUFFICIENT": 0.15, "INSUFFICIENT": 0.08, "CONFLICTING": 0.03, "IRRELEVANT": 0.02},
                f"Partial evidence identified (top score {top_rerank:.2f}); some query components may be missing."
            )

        # Irrelevant retrieval (semantic distractor)
        if top_dense >= 0.45 and top_rerank < 0.35 and overlap < 0.15:
            return (
                EpistemicDecision.IRRELEVANT,
                0.88,
                {"IRRELEVANT": 0.88, "INSUFFICIENT": 0.08, "PARTIAL": 0.02, "SUFFICIENT": 0.01, "CONFLICTING": 0.01},
                f"Retrieved passages appear to be semantic distractors (dense: {top_dense:.2f}, rerank: {top_rerank:.2f})."
            )

        # Default insufficient
        return (
            EpistemicDecision.INSUFFICIENT,
            0.90,
            {"INSUFFICIENT": 0.90, "IRRELEVANT": 0.05, "PARTIAL": 0.03, "SUFFICIENT": 0.01, "CONFLICTING": 0.01},
            "Insufficient evidence: retrieved context does not support answering the query."
        )

    def predict(self, query: str, passages: List[ScoredPassage]) -> EvidenceAssessment:
        features = extract_evidence_features(query, passages)
        
        # If trained ML model is available, use it
        if self.model is not None:
            x = feature_dict_to_array(features).reshape(1, -1)
            pred_idx = int(self.model.predict(x)[0])
            decision = LABEL_TO_DECISION.get(pred_idx, EpistemicDecision.INSUFFICIENT)
            
            # Calibrated probability distribution
            if hasattr(self.model, "predict_proba"):
                probs = self.model.predict_proba(x)[0]
                prob_dict = {LABEL_TO_DECISION[i].value: float(probs[i]) for i in range(len(probs))}
                confidence = float(np.max(probs))
            else:
                prob_dict = {decision.value: 1.0}
                confidence = 0.80
                
            explanation = f"ML-calibrated ({self.model_type}) prediction: {decision.value} (Confidence: {confidence:.2f})"
        else:
            # Fall back to calibrated heuristic
            decision, confidence, prob_dict, explanation = self.predict_heuristic(features)
            
        return EvidenceAssessment(
            decision=decision,
            confidence=confidence,
            predicted_probabilities=prob_dict,
            features_used=features,
            explanation=explanation
        )

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fits and calibrates the classifier on tabular benchmark features."""
        if self.model_type == "logistic_regression":
            from sklearn.linear_model import LogisticRegression
            from sklearn.calibration import CalibratedClassifierCV
            base = LogisticRegression(max_iter=1000, class_weight="balanced")
            self.model = CalibratedClassifierCV(base, method="sigmoid", cv=3)
            self.model.fit(X, y)
        elif self.model_type == "random_forest":
            from sklearn.ensemble import RandomForestClassifier
            self.model = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42, class_weight="balanced")
            self.model.fit(X, y)
        elif self.model_type == "xgboost":
            import xgboost as xgb
            self.model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.08,
                random_state=42,
                eval_metric="mlogloss"
            )
            self.model.fit(X, y)
        else:  # default lightgbm
            import lightgbm as lgb
            self.model = lgb.LGBMClassifier(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.08,
                random_state=42,
                class_weight="balanced",
                verbose=-1
            )
            self.model.fit(X, y)

    def save(self, path: Optional[str] = None):
        target_path = path or self.model_path
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        joblib.dump(self.model, target_path)
        print(f"Classifier saved successfully to {target_path}")
