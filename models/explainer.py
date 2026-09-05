"""SHAP explainability with merchant-readable reasons.

The consumer of a reason string is a merchant ops reviewer deciding whether to approve a
return, not a data scientist. "customer_return_rate_lt = 0.62 (SHAP +0.41)" is not
actionable; "Returns 62% of lifetime orders -- far above the 12% norm" is.

WHY SHAP RUNS ON THE RAW MODEL, NOT THE CALIBRATED SCORE
--------------------------------------------------------
`TreeExplainer` attributes the booster's raw log-odds margin. The Platt calibrator
applied on top (see models/train.py) is a STRICTLY monotone transform, so it cannot
change the sign or the ordering of any attribution -- the top-3 drivers of the calibrated
score are identically the top-3 drivers of the raw margin. Explaining the raw model keeps
the attributions exact and additive.

The API response documents `contribution` as raw-margin units, so nobody reads those
numbers as being on the same scale as `risk_score`.

DEFENSE-ONLY NOTE
-----------------
Reasons describe what *this* order looks like. They never state the decision boundary,
never say how far a value would have to move to flip the decision, and never rank which
signal is cheapest to defeat. See SAFETY.md.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
import shap

logger = logging.getLogger(__name__)

# Population reference points, used only to phrase a value as "above/below normal".
# These are descriptive anchors from the training distribution, not thresholds.
_NORM_RETURN_RATE = 0.12
_NORM_ACCOUNT_AGE = 400


def _fmt_inr(v: float) -> str:
    return f"INR {v:,.0f}"


def _reason_text(feature: str, value: float, increases_risk: bool) -> str:
    """Map (feature, value, direction) to a sentence a merchant ops reviewer can act on."""
    up = increases_risk
    if feature == "customer_return_rate_lt":
        pct = value * 100
        return (
            f"Returns {pct:.0f}% of lifetime orders, above the ~{_NORM_RETURN_RATE:.0%} norm"
            if up
            else f"Lifetime return rate of {pct:.0f}% is in line with normal customers"
        )
    if feature == "customer_return_rate_90d":
        pct = value * 100
        return (
            f"Recent 90-day return rate of {pct:.0f}% is elevated"
            if up
            else f"Recent 90-day return rate of {pct:.0f}% looks unremarkable"
        )
    if feature == "customer_account_age_days":
        if up:
            return f"Account is only {value:.0f} days old"
        return f"Established account ({value:.0f} days old, vs ~{_NORM_ACCOUNT_AGE} typical)"
    if feature == "customer_order_velocity_7d":
        return (
            f"{value:.0f} orders placed in the last 7 days -- unusually fast"
            if up
            else f"Ordering pace is normal ({value:.0f} orders in 7 days)"
        )
    if feature == "order_value":
        return (
            f"Order value of {_fmt_inr(value)} is high for this category"
            if up
            else f"Order value of {_fmt_inr(value)} is typical"
        )
    if feature == "discount_percentage":
        return (
            f"Purchased at a {value:.0f}% discount -- discount-abuse pattern"
            if up
            else f"Modest discount ({value:.0f}%)"
        )
    if feature == "category_return_base_rate":
        return (
            f"Category returns at a high base rate ({value:.0%})"
            if up
            else f"Category has a low baseline return rate ({value:.0%})"
        )
    if feature == "category_encoded":
        return "Product category carries above-average return-abuse risk" if up else \
               "Product category is low-risk for return abuse"
    if feature == "is_cod":
        if value >= 0.5:
            return "Cash-on-delivery order (higher RTO/return-loss exposure)" if up else \
                   "Cash-on-delivery, but other signals are clean"
        return "Prepaid order -- payment already captured" if not up else \
               "Prepaid order, but other signals are elevated"
    if feature == "days_to_return":
        if value < 0:
            return "Return window timing not available for this request"
        return (
            f"Return raised {value:.0f} days after delivery, late in the window"
            if up
            else f"Return raised promptly ({value:.0f} days after delivery)"
        )
    if feature == "same_address_returns_7d":
        return (
            f"{value:.0f} other returns from this address in 7 days -- possible ring"
            if up
            else "No unusual return clustering at this address"
        )
    if feature == "same_email_returns_7d":
        return (
            f"{value:.0f} other returns from this email in 7 days -- possible ring"
            if up
            else "No unusual return clustering on this email"
        )
    if feature == "payment_hash_collision":
        return "Payment instrument shared with other flagged accounts" if value >= 0.5 \
            else "Payment instrument not shared with other accounts"
    if feature == "ip_phone_collision":
        return "Device/phone fingerprint shared with other accounts" if value >= 0.5 \
            else "Device/phone fingerprint is unique to this account"
    if feature == "rto_risk_score":
        return (
            f"Delivery pincode has a high historical RTO rate ({value:.0%})"
            if up
            else f"Delivery pincode has a low historical RTO rate ({value:.0%})"
        )
    return f"{feature} = {value:.3g}"


class RiskExplainer:
    """Thin wrapper over `shap.TreeExplainer` producing merchant-readable reasons."""

    def __init__(self, model: Any) -> None:
        self.model = model
        self.explainer = shap.TreeExplainer(model)

    def shap_values(self, X: pd.DataFrame) -> np.ndarray:
        """SHAP values in the raw-margin space, shape (n_rows, n_features)."""
        vals = self.explainer.shap_values(X)
        vals = np.asarray(vals)
        # Binary XGBoost returns (n, f); guard against the (n, f, 2) layout some
        # shap/model combinations produce.
        if vals.ndim == 3:
            vals = vals[..., -1]
        return vals

    def top_reasons(
        self, X: pd.DataFrame, k: int = 3
    ) -> list[list[dict[str, Any]]]:
        """Top-k drivers per row, ranked by absolute contribution.

        Both risk-increasing and risk-decreasing drivers can appear: a reviewer needs to
        see why a borderline order was *not* flagged as much as why it was.
        """
        vals = self.shap_values(X)
        cols = list(X.columns)
        out: list[list[dict[str, Any]]] = []
        for i in range(len(X)):
            row = vals[i]
            idx = np.argsort(np.abs(row))[::-1][:k]
            reasons = []
            for j in idx:
                contribution = float(row[j])
                value = float(X.iloc[i, j])
                reasons.append(
                    {
                        "feature": cols[j],
                        "value": value,
                        "contribution": round(contribution, 4),
                        "direction": "increases_risk" if contribution > 0 else "decreases_risk",
                        "reason": _reason_text(cols[j], value, contribution > 0),
                    }
                )
            out.append(reasons)
        return out
