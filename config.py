"""Project-wide constants.

Single source of truth for the category taxonomy, the model's feature list, and the
cost assumptions. Imported by the generator, trainer, API and dashboard so that a
change here cannot drift between training and serving.
"""
from __future__ import annotations

# --- Category taxonomy (matches the API contract in api/schemas.py) ----------------
CATEGORIES: list[str] = ["Apparel", "Beauty", "Books", "Electronics", "Home"]

# Baseline *return* (not fraud) rate per category. These are the merchant-observable
# priors a real catalogue would supply; values are order-of-magnitude figures for
# Indian D2C (apparel returns dominate, books barely return).
CATEGORY_RETURN_BASE_RATE: dict[str, float] = {
    "Apparel": 0.26,
    "Beauty": 0.13,
    "Books": 0.06,
    "Electronics": 0.09,
    "Home": 0.11,
}

# Share of orders by category. Overridden at generation time by Olist-derived weights
# when data/raw/olist_order_items_dataset.csv is present (see data/real_calibration.py).
CATEGORY_ORDER_SHARE: dict[str, float] = {
    "Apparel": 0.34,
    "Beauty": 0.16,
    "Books": 0.09,
    "Electronics": 0.21,
    "Home": 0.20,
}

# Typical order value scale (INR) per category: gamma(shape, scale) + floor.
CATEGORY_VALUE_PARAMS: dict[str, tuple[float, float, float]] = {
    "Apparel": (2.0, 550.0, 250.0),
    "Beauty": (2.0, 380.0, 200.0),
    "Books": (1.8, 260.0, 120.0),
    "Electronics": (2.2, 2600.0, 700.0),
    "Home": (2.0, 900.0, 300.0),
}

# --- Model feature list -------------------------------------------------------------
# NOTE: `is_prepaid` is accepted by the API but deliberately NOT a model feature: it is
# definitionally 1 - is_cod. Feeding a perfectly collinear duplicate lets the booster
# split importance arbitrarily across the pair, which would weaken the per-feature
# importance cap enforced by the leakage test suite. The API keeps the field and
# validates the identity instead, so an inconsistent payload fails closed.
FEATURES: list[str] = [
    "customer_return_rate_lt",
    "customer_return_rate_90d",
    "customer_account_age_days",
    "customer_order_velocity_7d",
    "order_value",
    "discount_percentage",
    "category_encoded",
    "category_return_base_rate",
    "is_cod",
    "days_to_return",
    "same_address_returns_7d",
    "same_email_returns_7d",
    "payment_hash_collision",
    "ip_phone_collision",
    "rto_risk_score",
]

# --- Cost assumptions (ILLUSTRATIVE; see evaluate/sensitivity.py) -------------------
# FN: a fraudulent return approved. Unrecovered cost-of-goods on a mid-value order
#     (~ INR 380) + reverse logistics (~ INR 80) + support handling (~ INR 40).
# FP: a genuine return sent to manual review. ~8 min of an ops reviewer's time at a
#     fully-loaded INR 300/hr (~ INR 40) + a small goodwill/friction allowance.
# A real merchant must re-derive both from their own P&L; see evaluate/cost_curve.py.
COST_FN: float = 500.0
COST_FP: float = 50.0

RANDOM_SEED: int = 42
