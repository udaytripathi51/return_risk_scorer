"""Scoring service: model loading, fail-closed scoring, SHAP reasons.

FAIL-CLOSED IS THE WHOLE DESIGN
-------------------------------
There is exactly one way to reach `auto_approve`: a fully-parsed order whose model score
came back below the threshold. Every other path -- unrecognised category, a missing
model, an exception anywhere in feature assembly or inference -- returns
`risk_score = 1.0` and `manual_review`.

UNKNOWN CATEGORIES ARE NEVER GUESSED
------------------------------------
A naive encoder maps an unseen `category` to index 0, so it silently impersonates
whichever known category happens to sort first ("Apparel", here). That is worse than an
error: it produces a confident, wrong, *approvable* score for an input the model has no
basis to judge. Here an unknown category routes to review with an explicit reason.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
from typing import Any

import joblib
import numpy as np
import pandas as pd

from config import CATEGORIES, FEATURES
from models.explainer import RiskExplainer

logger = logging.getLogger(__name__)

MODEL_DIR = os.environ.get("MODEL_DIR", "models")

# A risk_score of exactly 1.0 means "the fail-closed path produced this", never "the model
# scored this at 1.0". Model-derived scores are clamped just below so the two are always
# distinguishable by a downstream consumer.
MAX_MODEL_SCORE = 0.999


def _fail_closed(
    reason: str, feature: str, threshold: float, model_version: str
) -> dict[str, Any]:
    """The single shape every non-model decision returns."""
    return {
        "risk_score": 1.0,
        "action": "manual_review",
        "reasons": [
            {
                "feature": feature,
                "value": None,
                "contribution": None,
                "direction": None,
                "reason": reason,
            }
        ],
        "threshold_used": float(threshold),
        "model_version": model_version,
        "data_quality_notes": ["Scored by the fail-closed path, not by the model."],
    }


class RiskScoringService:
    """Loads persisted artefacts and scores orders. Safe to construct when artefacts
    are missing: `is_ready()` returns False and every score fails closed."""

    def __init__(self, model_dir: str = MODEL_DIR) -> None:
        self.model_dir = model_dir
        self.model = None
        self.calibrator = None
        self.encoder = None
        self.explainer: RiskExplainer | None = None
        self.threshold: float = 0.5
        self.features: list[str] = list(FEATURES)
        self.category_names: list[str] = list(CATEGORIES)
        self.metadata: dict[str, Any] = {}
        self.model_version: str = "unloaded"
        self._load()

    # --- loading --------------------------------------------------------------------
    def _load(self) -> None:
        try:
            model_path = os.path.join(self.model_dir, "model.pkl")
            cal_path = os.path.join(self.model_dir, "calibrator.pkl")
            enc_path = os.path.join(self.model_dir, "encoder.pkl")
            meta_path = os.path.join(self.model_dir, "metadata.json")

            missing = [p for p in (model_path, enc_path, meta_path) if not os.path.exists(p)]
            if missing:
                logger.error("model artefacts missing: %s -- service will fail closed",
                             ", ".join(missing))
                return

            self.model = joblib.load(model_path)
            self.encoder = joblib.load(enc_path)
            self.calibrator = joblib.load(cal_path) if os.path.exists(cal_path) else None
            with open(meta_path, encoding="utf-8") as f:
                self.metadata = json.load(f)

            self.threshold = float(self.metadata.get("threshold", 0.5))
            self.features = list(self.metadata.get("features", FEATURES))
            self.category_names = list(self.metadata.get("categories", CATEGORIES))
            self.explainer = RiskExplainer(self.model)

            with open(model_path, "rb") as f:
                digest = hashlib.sha256(f.read()).hexdigest()[:12]
            self.model_version = f"rrs-{digest}"
            logger.info(
                "model loaded: version=%s threshold=%.4f features=%d",
                self.model_version, self.threshold, len(self.features),
            )
        except Exception:
            logger.exception("failed to load model artefacts -- service will fail closed")
            self.model = None
            self.explainer = None

    def is_ready(self) -> bool:
        return self.model is not None and self.encoder is not None and self.explainer is not None

    # --- feature assembly -----------------------------------------------------------
    def _to_frame(self, order: dict[str, Any]) -> tuple[pd.DataFrame, list[str]]:
        """Build the model input row. Returns (frame, data_quality_notes)."""
        notes: list[str] = []
        row = dict(order)
        row["category_encoded"] = int(self.encoder.transform([row["category"]])[0])

        # `days_to_return = -1` is the contract's "unknown" sentinel. Feeding -1 as a
        # number would place it below every value the model ever saw and let the tree
        # treat "unknown" as "returned extremely promptly", which reads as LOW risk --
        # precisely the wrong direction. Mapping it to NaN hands it to XGBoost's native
        # missing-value routing instead.
        if float(row.get("days_to_return", 0)) < 0:
            row["days_to_return"] = np.nan
            notes.append(
                "days_to_return was not supplied; scored with the model's missing-value "
                "path, which is lower-confidence than a fully-populated request."
            )

        frame = pd.DataFrame([{f: row.get(f) for f in self.features}], columns=self.features)
        return frame.astype(float), notes

    # --- scoring --------------------------------------------------------------------
    def score(self, order: dict[str, Any]) -> dict[str, Any]:
        if not self.is_ready():
            return _fail_closed(
                "Scoring model is not loaded — routed to manual review",
                "service", self.threshold, self.model_version,
            )

        category = order.get("category")
        # An unknown category fails closed instead of impersonating a known one.
        if category not in list(self.encoder.classes_):
            return _fail_closed(
                f'Unrecognized category "{category}" — routed to manual review',
                "category", self.threshold, self.model_version,
            )

        try:
            X, notes = self._to_frame(order)
            raw = float(self.model.predict_proba(X)[:, 1][0])
            score = float(self.calibrator.predict([raw])[0]) if self.calibrator else raw
            # 1.0 is RESERVED as the fail-closed sentinel. Clamping model-derived scores
            # just below it keeps "the model is certain this is fraud" distinguishable
            # from "we could not score this at all" -- two states that need very
            # different handling downstream, and which a shared 1.0 would conflate.
            score = float(np.clip(score, 0.0, MAX_MODEL_SCORE))
            reasons = self.explainer.top_reasons(X, k=3)[0]
            action = "manual_review" if score >= self.threshold else "auto_approve"
            return {
                "risk_score": round(score, 4),
                "action": action,
                "reasons": reasons,
                "threshold_used": float(self.threshold),
                "model_version": self.model_version,
                "data_quality_notes": notes,
            }
        except Exception:
            logger.exception("scoring failed — routing to manual review")
            return _fail_closed(
                "Scoring error — routed to manual review",
                "service", self.threshold, self.model_version,
            )

    def score_batch(self, orders: list[dict[str, Any]]) -> dict[str, Any]:
        """Score a batch. One bad order never poisons the rest -- each fails closed
        independently."""
        results = [self.score(o) for o in orders]
        flagged = sum(1 for r in results if r["action"] == "manual_review")
        scores = [r["risk_score"] for r in results]
        return {
            "results": results,
            "summary": {
                "n": len(results),
                "flagged_for_review": flagged,
                "auto_approved": len(results) - flagged,
                "flag_rate": round(flagged / max(len(results), 1), 4),
                "mean_risk_score": round(float(np.mean(scores)) if scores else 0.0, 4),
                "max_risk_score": round(float(np.max(scores)) if scores else 0.0, 4),
            },
        }
