"""Adversarial held-out slice: how the model holds up against an adapted abuser.

A stress slice is only meaningful if its label is not leaked either. Drawing `is_fraud`
first and then shifting `order_value` or `days_to_return` for the fraud rows would
measure how well the model reads that shift, not whether it generalises to fraud that
does not look like the training distribution. So this slice uses the same causal design
as the main generator.

WHAT THIS SLICE MODELS
----------------------
An *adapted* abuser: someone who has worked out roughly what gets flagged and is
deliberately staying inside normal-looking bounds. Concretely they
  - keep their observable return-rate history close to the legitimate population,
  - operate on aged accounts rather than fresh ones,
  - avoid shared-address / shared-email / device collisions almost entirely,
  - use prepaid at close to the legitimate rate rather than leaning on COD,
  - keep order values and discounts unremarkable,
  - and do not use throwaway ("burner") accounts, so the latent burner term of the label
    carries no signal here at all.

The construction is identical in *form* to data/synthetic.py -- a latent
`looks_genuine_fraudster` trait tilts the distributions, and the label is drawn from the
trait -- but every tilt is roughly a third to a half the size. Nothing is edited after
the label is realised.

The point of the slice is the RECALL GAP: recall here, at the exact same operating
threshold, versus recall on the in-distribution test set. A large gap is the honest
statement that the model leans on signals an adapted abuser can suppress. We report that
gap rather than hiding it -- it is the single most useful number for a merchant deciding
how much to trust the automated approve path.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from config import (
    CATEGORIES,
    CATEGORY_ORDER_SHARE,
    CATEGORY_RETURN_BASE_RATE,
    CATEGORY_VALUE_PARAMS,
)
from data.synthetic import DISCOUNT_LEVELS, N_PINCODES, _pincode_rto_rates

logger = logging.getLogger(__name__)

# Latent share of adapted abusers in this slice. Higher than the population rate because
# the slice is deliberately enriched -- it is a stress test, not a sample of production.
ADAPTED_RATE = 0.15

# Discount mix for an adapted abuser: nudged toward higher discounts, but far closer to
# the legitimate profile than the main generator's fraudster mix.
DISCOUNT_P_ADAPTED = np.array([0.17, 0.23, 0.20, 0.16, 0.12, 0.08, 0.04])
DISCOUNT_P_LEGIT = np.array([0.20, 0.25, 0.20, 0.15, 0.10, 0.07, 0.03])


def generate_adversarial(
    n_rows: int = 12_000,
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    """One row per return request, drawn from the adapted-abuser distribution."""
    rng = rng or np.random.default_rng(2026)
    n = n_rows

    # --- Latent trait, decided first ------------------------------------------------
    adapted = rng.random(n) < ADAPTED_RATE

    # Burner accounts: an adapted abuser specifically does NOT use throwaway accounts --
    # that is one of the cheapest signals to defeat. Their burner rate is at or below the
    # legitimate population's, so this term of the label carries no signal in this slice.
    is_burner = rng.random(n) < np.where(adapted, 0.04, 0.05)

    # Account age: adapted abusers age their accounts on purpose, so the tilt is small
    # and in the *opposite* direction to the naive fraudster (older, not younger).
    age = np.where(
        is_burner,
        rng.gamma(1.1, 22.0, n),
        np.where(adapted, rng.gamma(2.3, 240.0, n), rng.gamma(2.5, 250.0, n)),
    )
    account_age_days = np.clip(age, 1, 3000).astype(int)

    # Return-rate history: kept deliberately close to legitimate. Mean ~0.16 vs ~0.12,
    # against ~0.33 for the naive fraudster in data/synthetic.py.
    propensity = np.where(adapted, rng.beta(2.6, 14.0, n), rng.beta(2.2, 16.0, n))
    n_hist = 1 + rng.poisson(np.clip(account_age_days / 120.0, 0.3, 25.0))
    return_rate_lt = rng.binomial(n_hist, propensity) / n_hist
    n_90d = 1 + rng.poisson(1.8, n)
    return_rate_90d = rng.binomial(n_90d, propensity) / n_90d

    velocity = rng.poisson(np.where(adapted, 1.1, 0.9))

    # Ring collisions almost entirely suppressed: the adapted abuser uses distinct
    # addresses, emails and devices. This is the signal they can most cheaply defeat.
    in_ring = rng.random(n) < np.where(adapted, 0.06, 0.03)

    n_high_rto = int(N_PINCODES * 0.18)
    from_high = rng.random(n) < np.where(adapted, 0.22, 0.17)
    pincode = np.where(
        from_high,
        rng.integers(0, n_high_rto, n),
        rng.integers(n_high_rto, N_PINCODES, n),
    )
    rto_by_pincode = _pincode_rto_rates(rng)

    # --- Observable features --------------------------------------------------------
    cat_p = np.array([CATEGORY_ORDER_SHARE[c] for c in CATEGORIES], dtype=float)
    cat_p = cat_p / cat_p.sum()
    cat_idx = rng.choice(len(CATEGORIES), size=n, p=cat_p)

    shape = np.array([CATEGORY_VALUE_PARAMS[c][0] for c in CATEGORIES])[cat_idx]
    scale = np.array([CATEGORY_VALUE_PARAMS[c][1] for c in CATEGORIES])[cat_idx]
    floor = np.array([CATEGORY_VALUE_PARAMS[c][2] for c in CATEGORIES])[cat_idx]
    tilt = np.where(adapted, 1.05, 1.0)          # vs 1.18 in the naive slice

    u = rng.random(n)
    cum_a = np.cumsum(DISCOUNT_P_ADAPTED)
    cum_l = np.cumsum(DISCOUNT_P_LEGIT)
    pick = np.where(
        adapted,
        np.searchsorted(cum_a, u, side="right"),
        np.searchsorted(cum_l, u, side="right"),
    )

    is_cod = (rng.random(n) < np.where(adapted, 0.44, 0.40)).astype(int)  # vs 0.55

    df = pd.DataFrame(
        {
            "customer_id": np.arange(n),
            "customer_account_age_days": account_age_days,
            "customer_return_rate_lt": return_rate_lt,
            "customer_return_rate_90d": return_rate_90d,
            "customer_order_velocity_7d": velocity,
            "category": np.array(CATEGORIES, dtype=object)[cat_idx],
            "category_return_base_rate": np.array(
                [CATEGORY_RETURN_BASE_RATE[c] for c in CATEGORIES]
            )[cat_idx],
            "order_value": np.clip(rng.gamma(shape, scale * tilt) + floor, 100, 50_000),
            "discount_percentage": DISCOUNT_LEVELS[
                np.clip(pick, 0, len(DISCOUNT_LEVELS) - 1)
            ],
            "is_cod": is_cod,
            "is_prepaid": 1 - is_cod,
            "same_address_returns_7d": rng.poisson(np.where(in_ring, 2.5, 0.15)),
            "same_email_returns_7d": rng.poisson(np.where(in_ring, 1.8, 0.10)),
            "payment_hash_collision": (
                rng.random(n) < np.where(in_ring, 0.30, 0.03)
            ).astype(int),
            "ip_phone_collision": (
                rng.random(n) < np.where(in_ring, 0.28, 0.025)
            ).astype(int),
            "rto_risk_score": rto_by_pincode[pincode],
            # Days to return: nudged later, but nowhere near the naive profile.
            "days_to_return": np.round(
                np.clip(
                    np.where(adapted, rng.gamma(3.8, 2.1, n), rng.gamma(3.2, 2.1, n)),
                    0,
                    45,
                ),
                1,
            ),
        }
    )

    # --- Label, from the latent trait only ------------------------------------------
    # Same functional form as data/synthetic.py. No feature is rewritten after this line.
    fraud_prob = np.minimum(
        0.55 * adapted.astype(float) + 0.08 * is_burner.astype(float) + 0.04, 0.95
    )
    df["is_fraud"] = (rng.random(n) < fraud_prob).astype(int)
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    d = generate_adversarial()
    print(
        {
            "n": len(d),
            "fraud_rate": round(float(d["is_fraud"].mean()), 4),
            "mean_return_rate_lt": round(float(d["customer_return_rate_lt"].mean()), 4),
            "collision_rate": round(float(d["payment_hash_collision"].mean()), 4),
        }
    )
