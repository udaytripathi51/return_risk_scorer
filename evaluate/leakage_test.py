"""Automated label-leakage smell test (PROJECT_SPEC.md section 7.1).

This is the check that makes the section 6.1 fix permanent and visible. It runs
standalone (`python -m evaluate.leakage_test`), as a pytest (`tests/test_leakage.py`),
and in CI on every push -- so a future change that reintroduces leakage fails the build
rather than quietly inflating a headline metric.

THREE CHECKS
------------
1. CONCENTRATION. No single feature may hold more than 25% of total gain importance.
   A leaked label shows up as one feature swallowing the model: in the pre-fix version,
   the order-value/COD/discount cluster dominated because the label was a function of
   them.

2. TOP-3 ABLATION. Drop the three most important features, retrain from scratch, and
   require that PR-AUC falls by no more than 50% relative. A model reading a leaked
   label formula collapses to near-baseline the moment those columns go (the pre-fix
   model went ~0.95 -> ~0.32, a 66% relative fall). A model aggregating genuinely
   distributed evidence degrades gracefully instead.

3. LABEL-SHUFFLE CONTROL. Retrain on permuted labels and require PR-AUC to collapse to
   the no-skill baseline (the base rate). This is a control for the *pipeline* rather
   than the data: if the split, encoding or calibration plumbing leaked information,
   the shuffled model would still score above baseline. It should not.

WHY 25% AND 50%
---------------
Both are the thresholds named in the spec. They are deliberately loose -- they are
tripwires for the failure mode that actually occurred, not a claim that a model at 24%
concentration is well-behaved. They are asserted against, never tuned to fit: when the
first build came in at 27% and the second at 29.9%, the generator's account-age
assumption was corrected on realism grounds, not the threshold.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any

import numpy as np
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


def run_leakage_checks(
    n_customers: int = 40_000,
    seed: int = RANDOM_SEED,
    verbose: bool = True,
) -> dict[str, Any]:
    """Train, ablate and shuffle. Returns a full result dict; raises nothing."""
    from data.synthetic import generate_dataset

    rng = np.random.default_rng(seed)
    df = generate_dataset(n_customers=n_customers, rng=rng)
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
        _print_report(result)
    return result


def _print_report(r: dict[str, Any]) -> None:
    def mark(ok: bool) -> str:
        return "PASS" if ok else "FAIL"

    c, a, s = r["concentration"], r["top3_ablation"], r["label_shuffle_control"]
    print("\n=== Leakage smell test " + "=" * 47)
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
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    res = run_leakage_checks(n_customers=60_000)
    os.makedirs("outputs", exist_ok=True)
    with open("outputs/leakage_report.json", "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    print("wrote outputs/leakage_report.json")
    raise SystemExit(0 if res["all_passed"] else 1)
