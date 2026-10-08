"""Automated label-leakage test suite.

Label leakage is the failure that makes a fraud model look brilliant offline and useless in
production: when a feature is effectively part of the label, the model reconstructs the
label instead of learning fraud. This suite turns "no leakage" from a claim into a tested
property. It runs standalone (`python -m evaluate.leakage_test`), as a pytest
(`tests/test_leakage.py`) and in CI on every push, so a change that introduces leakage
fails the build instead of quietly inflating a headline metric.

THREE CHECKS
------------
1. CONCENTRATION. No single feature may hold more than 25% of total gain importance. A
   leaked label shows up as one feature, or a small cluster, swallowing the model.

2. TOP-3 ABLATION. Drop the three most important features, retrain from scratch, and
   require that PR-AUC falls by no more than 50% relative. A model reading a leaked label
   formula collapses towards the base rate once those columns go; a model aggregating
   genuinely distributed evidence degrades gracefully.

3. LABEL-SHUFFLE CONTROL. Retrain on permuted labels and require PR-AUC to collapse to the
   no-skill baseline (the base rate). This is a negative control for the train/evaluate
   harness: with permuted labels there is nothing to learn, so a score above the base rate
   would expose a harness defect -- scoring rows the model trained on, or test labels
   reaching model selection. It cannot detect a feature computed from the label (permuting
   breaks that relationship too); checks 1 and 2 target that.
   Calibration is not exercised here: raw probabilities are used throughout.

POSITIVE CONTROL: THE SUITE MUST BE ABLE TO FAIL
------------------------------------------------
A test that cannot fail proves nothing when it passes. `make_leaky_control()` builds a
deliberately leaky copy of the data -- the label recomputed from four observable features,
which are then pushed further out for the fraud rows, the textbook way a synthetic fraud
dataset leaks -- and the suite is required to flag it. CI checks both directions: the real
data passes and the leaky control fails.

WHY 25% AND 50%
---------------
They are deliberately loose tripwires for the failure mode above, not a claim that a model
at 24% concentration is well-behaved. They are asserted against, never tuned to fit: the
generator's design (`data/synthetic.py`) is what keeps the real model inside them.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from config import RANDOM_SEED
from models.train import build_encoder, fit_model, prepare_features, split_by_customer

logger = logging.getLogger(__name__)

MAX_SINGLE_FEATURE_IMPORTANCE = 0.25
MAX_RELATIVE_PR_AUC_DROP = 0.50
# A shuffled-label model must not beat the base rate by more than this multiple.
MAX_SHUFFLE_LIFT = 1.25


def _pr_auc(model, X, y) -> float:
    return float(average_precision_score(y, model.predict_proba(X)[:, 1]))


def make_leaky_control(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """A deliberately leaky copy of `df`: the positive control for this suite.

    The label is recomputed FROM four observable features -- order value, the COD flag,
    discount and days to return -- and those same features are then pushed further out
    for the rows that came up fraudulent. That is the textbook way a synthetic fraud
    dataset leaks its label, and every check above should flag it. Used only by the
    tests and the standalone run; never by training.
    """
    out = df.copy()
    log_value = np.log(out["order_value"])
    z = (
        0.9 * (log_value - log_value.mean()) / log_value.std()
        + 0.8 * out["is_cod"]
        + 0.04 * out["discount_percentage"]
        + 0.12 * out["days_to_return"]
        - 4.4
    )
    y = (rng.random(len(out)) < 1.0 / (1.0 + np.exp(-z))).astype(int)
    fraud = y == 1
    out["is_fraud"] = y
    out.loc[fraud, "order_value"] = out.loc[fraud, "order_value"] * 1.5
    out.loc[fraud, "discount_percentage"] = np.minimum(
        out.loc[fraud, "discount_percentage"] + 20, 90
    )
    out.loc[fraud, "days_to_return"] = out.loc[fraud, "days_to_return"] + 6
    return out


def run_leakage_checks(
    n_customers: int = 40_000,
    seed: int = RANDOM_SEED,
    verbose: bool = True,
    df: pd.DataFrame | None = None,
    title: str = "Leakage test suite",
) -> dict[str, Any]:
    """Train, ablate and shuffle. Returns a full result dict; raises nothing.

    Runs on a freshly generated dataset unless `df` is given (the positive control passes
    its leaky copy here).
    """
    from data.synthetic import generate_dataset

    if df is None:
        df = generate_dataset(n_customers=n_customers, rng=np.random.default_rng(seed))
    encoder = build_encoder()
    train_df, val_df, test_df = split_by_customer(df, seed=seed)

    X_train = prepare_features(train_df, encoder)
    X_val = prepare_features(val_df, encoder)
    X_test = prepare_features(test_df, encoder)
    y_train = train_df["is_fraud"].to_numpy()
    y_val = val_df["is_fraud"].to_numpy()
    y_test = test_df["is_fraud"].to_numpy()

    base_rate = float(y_test.mean())

    # --- Check 1: importance concentration ------------------------------------------
    # Raw model probabilities are used throughout: Platt calibration is strictly monotone
    # so it cannot change PR-AUC, and leaving it out keeps this test independent of the
    # calibration step entirely.
    model = fit_model(X_train, y_train, X_val, y_val, seed=seed)
    importances = np.asarray(model.feature_importances_, dtype=float)
    features = list(X_train.columns)
    total = float(importances.sum()) or 1.0
    shares = importances / total

    order = np.argsort(shares)[::-1]
    ranked = [(features[i], round(float(shares[i]), 4)) for i in order]
    max_share = float(shares.max())
    check1_pass = bool(max_share <= MAX_SINGLE_FEATURE_IMPORTANCE)

    full_pr_auc = _pr_auc(model, X_test, y_test)

    # --- Check 2: drop the top 3, retrain -------------------------------------------
    top3 = [features[i] for i in order[:3]]
    keep = [f for f in features if f not in top3]
    reduced = fit_model(X_train[keep], y_train, X_val[keep], y_val, seed=seed)
    reduced_pr_auc = _pr_auc(reduced, X_test[keep], y_test)

    rel_drop = (full_pr_auc - reduced_pr_auc) / full_pr_auc if full_pr_auc > 0 else 1.0
    check2_pass = bool(rel_drop <= MAX_RELATIVE_PR_AUC_DROP)

    # --- Check 3: label-shuffle control ---------------------------------------------
    shuffle_rng = np.random.default_rng(seed + 1)
    y_train_shuf = shuffle_rng.permutation(y_train)
    y_val_shuf = shuffle_rng.permutation(y_val)
    shuffled = fit_model(X_train, y_train_shuf, X_val, y_val_shuf, seed=seed)
    shuffled_pr_auc = _pr_auc(shuffled, X_test, y_test)
    shuffle_lift = shuffled_pr_auc / base_rate if base_rate > 0 else float("inf")
    check3_pass = bool(shuffle_lift <= MAX_SHUFFLE_LIFT)

    result: dict[str, Any] = {
        "n_customers": n_customers,
        "n_return_requests": int(len(df)),
        "test_base_rate": round(base_rate, 4),
        "concentration": {
            "max_single_feature_importance": round(max_share, 4),
            "cap": MAX_SINGLE_FEATURE_IMPORTANCE,
            "top_feature": ranked[0][0],
            "ranked_importances": ranked,
            "passed": check1_pass,
        },
        "top3_ablation": {
            "dropped_features": top3,
            "full_pr_auc": round(full_pr_auc, 4),
            "reduced_pr_auc": round(reduced_pr_auc, 4),
            "relative_drop": round(float(rel_drop), 4),
            "max_allowed_relative_drop": MAX_RELATIVE_PR_AUC_DROP,
            "passed": check2_pass,
        },
        "label_shuffle_control": {
            "shuffled_pr_auc": round(shuffled_pr_auc, 4),
            "base_rate": round(base_rate, 4),
            "lift_over_base_rate": round(float(shuffle_lift), 4),
            "max_allowed_lift": MAX_SHUFFLE_LIFT,
            "passed": check3_pass,
        },
    }
    result["all_passed"] = bool(check1_pass and check2_pass and check3_pass)

    if verbose:
        _print_report(result, title)
    return result


def _print_report(r: dict[str, Any], title: str = "Leakage test suite") -> None:
    def mark(ok: bool) -> str:
        return "PASS" if ok else "FAIL"

    c, a, s = r["concentration"], r["top3_ablation"], r["label_shuffle_control"]
    print(f"\n=== {title} " + "=" * max(4, 66 - len(title)))
    print(f"dataset: {r['n_return_requests']} return requests, "
          f"base rate {r['test_base_rate']:.4f}\n")

    print(f"[{mark(c['passed'])}] 1. Importance concentration")
    print(f"        max single feature = {c['max_single_feature_importance']:.1%} "
          f"({c['top_feature']}), cap {c['cap']:.0%}")
    for name, share in c["ranked_importances"][:5]:
        print(f"          {name:32s} {share:6.1%}")

    print(f"\n[{mark(a['passed'])}] 2. Top-3 ablation")
    print(f"        dropped: {', '.join(a['dropped_features'])}")
    print(f"        PR-AUC {a['full_pr_auc']:.4f} -> {a['reduced_pr_auc']:.4f} "
          f"({a['relative_drop']:.1%} relative drop, "
          f"max {a['max_allowed_relative_drop']:.0%})")

    print(f"\n[{mark(s['passed'])}] 3. Label-shuffle control")
    print(f"        shuffled PR-AUC {s['shuffled_pr_auc']:.4f} vs base rate "
          f"{s['base_rate']:.4f} (lift {s['lift_over_base_rate']:.2f}x, "
          f"max {s['max_allowed_lift']}x)")
    print("\n" + "=" * 70)
    print("RESULT:", "ALL CHECKS PASSED" if r["all_passed"] else "LEAKAGE SUSPECTED")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    from data.synthetic import generate_dataset

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    res = run_leakage_checks(n_customers=60_000, title="Leakage test suite: project data")

    leaky = make_leaky_control(
        generate_dataset(n_customers=60_000, rng=np.random.default_rng(RANDOM_SEED)),
        rng=np.random.default_rng(RANDOM_SEED + 2),
    )
    control = run_leakage_checks(
        n_customers=60_000, df=leaky, title="Positive control: deliberately leaky data"
    )
    caught = not control["all_passed"]
    print("Positive control:", "leak detected, as required" if caught
          else "LEAK NOT DETECTED -- the suite has lost its teeth")

    res["positive_control"] = control
    os.makedirs("outputs", exist_ok=True)
    with open("outputs/leakage_report.json", "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    print("wrote outputs/leakage_report.json")
    raise SystemExit(0 if res["all_passed"] and caught else 1)
