"""Corrected synthetic generator for return-fraud data (PROJECT_SPEC.md section 6.1).

THE BUG THIS FIXES
------------------
The previous generator built `fraud_prob` from `order_value`, `is_cod`,
`discount_percentage` and `days_to_return`, drew `is_fraud` from it, and *then* pushed
those same features further out for the rows that came up fraudulent. A derived
`rto_risk_score` re-combined the same signals a third time. The model therefore learned
to invert the label formula rather than to recognise fraud, and PR-AUC of ~0.95
collapsed to ~0.32 the moment those features were withheld.

THE CAUSAL STRUCTURE USED INSTEAD
---------------------------------
    latent customer trait  ->  tilts the distributions features are drawn from
                           ->  observable features
                           ->  (independently) the label

    is_fraudster, is_burner  (latent, never columns in the dataset)
        |-> account age, return-rate history, order velocity, pincode, ring membership
        |-> per-order: COD share, order value, discount, days-to-return
        `-> fraud_prob = 0.55*is_fraudster + 0.08*is_burner + 0.04

WHY THE LABEL USES A LATENT `is_burner` AND NOT `account_age < 30`
------------------------------------------------------------------
PROJECT_SPEC.md section 6.1 sketches the label as
`0.55*is_fraudster + 0.08*(account_age < 30) + 0.04`. That sketch still puts an
*observed model feature* (account age) directly inside the label formula, which
contradicts the principle the same section is there to enforce: no feature used to
construct the label may also be fed to the model. It also plants a sharp discontinuity
at exactly 30 days that a tree ensemble will find and exploit -- when we built it that
way, `customer_account_age_days` took 27% of total gain importance, over the 25% cap
the leakage test in section 7.1 enforces.

So the age term is replaced by a second LATENT trait, `is_burner` (a throwaway account
opened to run an abuse cycle). The burner trait tilts observed account age sharply
downward, so age stays genuinely predictive -- but only as a noisy proxy for an
unobserved cause, with no exact threshold to reverse-engineer. Every term in the label
is now latent. Section 0.1 explicitly permits this: the principle is binding, the
example code is not.

No feature is touched after `is_fraud` is realised. Every tilt is a shift in a
distribution with substantial overlap, never a threshold or a deterministic rule, so a
model cannot recover the label formula: it can only estimate the latent trait from noisy
evidence. The 4% floor in `fraud_prob` (a genuine return that later gets charged back or
disputed) and the 0.55 ceiling on the trait's contribution are what cap achievable
PR-AUC in the 0.3-0.5 band -- that irreducible label noise is the point, not a defect.

`rto_risk_score` is NOT a recombination of the other features here. It is the historical
RTO rate of the delivery pincode -- a geography-level statistic a merchant reads off its
own logistics history. It carries mild signal only because fraud rings cluster
geographically (fraudsters are tilted toward high-RTO pincodes), which is a property of
the latent trait, exactly like every other feature.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from config import (
    CATEGORIES,
    CATEGORY_ORDER_SHARE,
    CATEGORY_RETURN_BASE_RATE,
    CATEGORY_VALUE_PARAMS,
    RANDOM_SEED,
)

logger = logging.getLogger(__name__)

# --- Population parameters ----------------------------------------------------------
# Share of the customer base that is a habitual return-abuser. Chosen once, on the
# order-of-magnitude figures merchants publicly quote for serial return abuse (low
# single-digit percent of accounts). It is NOT tuned against any metric.
FRAUDSTER_RATE = 0.06
# Share of fraudsters operating as part of a ring (shared address / email / device).
RING_SHARE_FRAUD = 0.35
# Share of legitimate customers who also trip collision signals -- shared family
# addresses, office deliveries, a household on one payment card. Without this the
# network features become near-perfect separators, which is not how they behave in
# production.
RING_SHARE_LEGIT = 0.03

DISCOUNT_LEVELS = np.array([0, 5, 10, 20, 30, 50, 70], dtype=float)
DISCOUNT_P_FRAUD = np.array([0.10, 0.15, 0.15, 0.20, 0.15, 0.15, 0.10])
DISCOUNT_P_LEGIT = np.array([0.20, 0.25, 0.20, 0.15, 0.10, 0.07, 0.03])

N_PINCODES = 600


def _draw_customers(n: int, rng: np.random.Generator) -> pd.DataFrame:
    """Customer-level attributes. Everything here is tilted by the latent trait."""
    is_fraudster = rng.random(n) < FRAUDSTER_RATE

    # Second latent trait: a throwaway / "burner" account opened to run an abuse cycle.
    # See the module docstring for why this exists rather than an `account_age < 30`
    # term in the label.
    is_burner = rng.random(n) < np.where(is_fraudster, 0.30, 0.05)

    # Account age. Outside the burner sub-population, a habitual abuser's account is only
    # mildly younger than a legitimate one (mean ~440d vs ~625d, with the bulk of both
    # distributions overlapping). This is deliberate realism: serial return abuse is very
    # often run from long-tenured accounts in good standing, which is exactly what makes
    # it expensive to catch. Burners are days-to-weeks old, so observed age is a NOISY
    # PROXY for the burner trait rather than a restatement of it.
    age = np.where(
        is_burner,
        rng.gamma(1.1, 22.0, n),
        np.where(is_fraudster, rng.gamma(2.2, 200.0, n), rng.gamma(2.5, 250.0, n)),
    )
    account_age_days = np.clip(age, 1, 3000).astype(int)

    # Latent per-customer propensity to return. Drives both the observed history and the
    # actual return events, which is what makes the history genuinely predictive without
    # being a restatement of the label.
    return_propensity = np.where(
        is_fraudster,
        rng.beta(4.0, 8.0, n),      # mean ~0.33
        rng.beta(2.2, 16.0, n),     # mean ~0.12
    )

    # Observed return-rate history = a NOISY, finite-sample estimate of the propensity.
    # Short-tenure customers have few prior orders, so their observed rate is noisy --
    # this is why the model cannot simply read the propensity off the feature. An
    # established account accumulates enough orders for a reasonably tight estimate,
    # which is why return history, not tenure, is the signal a merchant actually leans on.
    n_hist = 2 + rng.poisson(np.clip(account_age_days / 60.0, 0.5, 40.0))
    return_rate_lt = rng.binomial(n_hist, return_propensity) / n_hist

    n_90d = 1 + rng.poisson(1.8, n)
    return_rate_90d = rng.binomial(n_90d, return_propensity) / n_90d

    velocity = rng.poisson(np.where(is_fraudster, 1.8, 0.9))

    # Ring membership: a latent structural property, decided before any order exists.
    in_ring = rng.random(n) < np.where(is_fraudster, RING_SHARE_FRAUD, RING_SHARE_LEGIT)

    # Pincode: fraud rings cluster in a minority of pincodes. Pincodes 0..n_high-1 are
    # the historically high-RTO ones.
    n_high_rto = int(N_PINCODES * 0.18)
    from_high = rng.random(n) < np.where(is_fraudster, 0.42, 0.17)
    pincode = np.where(
        from_high,
        rng.integers(0, n_high_rto, n),
        rng.integers(n_high_rto, N_PINCODES, n),
    )

    return pd.DataFrame(
        {
            "customer_id": np.arange(n),
            "_is_fraudster": is_fraudster,            # latent: dropped before export
            "_is_burner": is_burner,                  # latent: dropped before export
            "_return_propensity": return_propensity,  # latent: dropped before export
            "_in_ring": in_ring,                      # latent: dropped before export
            "customer_account_age_days": account_age_days,
            "customer_return_rate_lt": return_rate_lt,
            "customer_return_rate_90d": return_rate_90d,
            "customer_order_velocity_7d": velocity,
            "pincode": pincode,
        }
    )


def _pincode_rto_rates(rng: np.random.Generator) -> np.ndarray:
    """Historical RTO rate per pincode -- an independent, geography-level statistic.

    Deliberately NOT a function of order_value / is_cod / discount. High-RTO pincodes
    (dense clusters, addresses with poor serviceability) get a materially higher base
    rate; the rest sit low. This is the merchant's own delivery history.
    """
    n_high = int(N_PINCODES * 0.18)
    rates = np.empty(N_PINCODES)
    rates[:n_high] = np.clip(rng.beta(6.0, 12.0, n_high), 0.02, 0.95)   # mean ~0.33
    rates[n_high:] = np.clip(rng.beta(3.0, 24.0, N_PINCODES - n_high), 0.02, 0.95)
    return rates


def generate_dataset(
    n_customers: int = 60_000,
    rng: np.random.Generator | None = None,
    category_share: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Generate the return-request dataset the model is trained and scored on.

    Returns one row per *return request* (the population the API actually sees), with
    `is_fraud` as the label. Latent columns are dropped before returning.
    """
    rng = rng or np.random.default_rng(RANDOM_SEED)
    share = category_share or CATEGORY_ORDER_SHARE
    cat_p = np.array([share[c] for c in CATEGORIES], dtype=float)
    cat_p = cat_p / cat_p.sum()

    cust = _draw_customers(n_customers, rng)
    rto_by_pincode = _pincode_rto_rates(rng)

    # Orders per customer, then explode to order level.
    n_orders = 1 + rng.poisson(2.5, n_customers)
    idx = np.repeat(cust.index.to_numpy(), n_orders)
    o = cust.loc[idx].reset_index(drop=True)
    n = len(o)
    logger.info("generated %d orders across %d customers", n, n_customers)

    fraudster = o["_is_fraudster"].to_numpy()
    burner = o["_is_burner"].to_numpy()
    in_ring = o["_in_ring"].to_numpy()

    # --- Order-level features. Every draw is tilted by the latent trait only. --------
    cat_idx = rng.choice(len(CATEGORIES), size=n, p=cat_p)
    o["category"] = np.array(CATEGORIES, dtype=object)[cat_idx]
    o["category_return_base_rate"] = np.array(
        [CATEGORY_RETURN_BASE_RATE[c] for c in CATEGORIES]
    )[cat_idx]

    # Order value: category sets the scale; the trait applies a mild multiplicative tilt
    # (abusers skew slightly higher-value, but the distributions overlap almost fully).
    shape = np.array([CATEGORY_VALUE_PARAMS[c][0] for c in CATEGORIES])[cat_idx]
    scale = np.array([CATEGORY_VALUE_PARAMS[c][1] for c in CATEGORIES])[cat_idx]
    floor = np.array([CATEGORY_VALUE_PARAMS[c][2] for c in CATEGORIES])[cat_idx]
    tilt = np.where(fraudster, 1.18, 1.0)
    o["order_value"] = np.clip(rng.gamma(shape, scale * tilt) + floor, 100, 50_000)

    # Discount: tilted probability vectors, not a rule.
    u = rng.random(n)
    cum_f = np.cumsum(DISCOUNT_P_FRAUD)
    cum_l = np.cumsum(DISCOUNT_P_LEGIT)
    pick = np.where(
        fraudster,
        np.searchsorted(cum_f, u, side="right"),
        np.searchsorted(cum_l, u, side="right"),
    )
    o["discount_percentage"] = DISCOUNT_LEVELS[np.clip(pick, 0, len(DISCOUNT_LEVELS) - 1)]

    # COD share: India-specific. Tilted, not deterministic.
    o["is_cod"] = (rng.random(n) < np.where(fraudster, 0.55, 0.40)).astype(int)
    o["is_prepaid"] = 1 - o["is_cod"]

    # Network / collision features come from ring membership (a latent structural
    # property), never from the label.
    o["same_address_returns_7d"] = rng.poisson(np.where(in_ring, 2.5, 0.15))
    o["same_email_returns_7d"] = rng.poisson(np.where(in_ring, 1.8, 0.10))
    o["payment_hash_collision"] = (rng.random(n) < np.where(in_ring, 0.30, 0.03)).astype(int)
    o["ip_phone_collision"] = (rng.random(n) < np.where(in_ring, 0.28, 0.025)).astype(int)

    # Independent geography signal.
    o["rto_risk_score"] = rto_by_pincode[o["pincode"].to_numpy()]

    # --- Return event ---------------------------------------------------------------
    # Whether the order comes back at all: the customer's latent propensity scaled by how
    # returnable the category is. Independent of the fraud draw below.
    cat_mult = o["category_return_base_rate"].to_numpy() / 0.13
    p_return = np.clip(o["_return_propensity"].to_numpy() * cat_mult, 0.005, 0.95)
    is_returned = rng.random(n) < p_return

    # Days to return: abusers sit somewhat closer to the end of the return window (they
    # use the item first), but the distributions overlap heavily -- mean ~8.8d vs ~6.7d.
    # An earlier draft used gamma(6.0, 2.2) here, a ~2x separation in means. That made a
    # single order-level timing feature the strongest predictor in the model (19% of gain,
    # 0.30 correlation with the label), which is not how return timing behaves in
    # production: it is weak, noisy evidence. The tilt is now sized to match that.
    days = np.where(fraudster, rng.gamma(4.2, 2.1, n), rng.gamma(3.2, 2.1, n))
    o["days_to_return"] = np.round(np.clip(days, 0, 45), 1)

    # --- Label ----------------------------------------------------------------------
    # Derived from the LATENT trait plus mild independent noise. Nothing above is
    # rewritten after this point -- that rewrite was the original leak.
    fraud_prob = np.minimum(
        0.55 * fraudster.astype(float) + 0.08 * burner.astype(float) + 0.04,
        0.95,
    )
    o["is_fraud"] = (is_returned & (rng.random(n) < fraud_prob)).astype(int)

    # The API scores return *requests*, so the modelling population is returned orders.
    out = o.loc[is_returned].reset_index(drop=True)
    out = out.drop(
        columns=[
            "_is_fraudster", "_is_burner", "_return_propensity", "_in_ring", "pincode",
        ]
    )
    return out


def summarise(df: pd.DataFrame) -> dict[str, Any]:
    return {
        "n_return_requests": int(len(df)),
        "fraud_rate": round(float(df["is_fraud"].mean()), 4),
        "n_fraud": int(df["is_fraud"].sum()),
        "cod_share": round(float(df["is_cod"].mean()), 4),
        "mean_order_value": round(float(df["order_value"].mean()), 2),
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    d = generate_dataset()
    print(summarise(d))
    print(d.head())
