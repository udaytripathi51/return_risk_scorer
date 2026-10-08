"""Diagnostics that answer the "compared to what?" questions about the headline numbers.

    python -m evaluate.diagnostics              # about a minute on a laptop CPU
    python -m evaluate.diagnostics --seeds 8    # wider seed sweep

Writes outputs/diagnostics.json. Deliberately NOT part of run.py: these are sanity checks
around the headline numbers, not the headline numbers themselves.

1. CEILING. The label is drawn from latent traits the model never sees, so even a perfect
   model is capped. An oracle that knows the traits scores P(fraud | traits) exactly; its
   PR-AUC is the best any model can reach on this data.
2. BASELINES. Same split, same features: logistic regression, a random forest, an
   unweighted XGBoost, and a three-rule policy. Answers "is the model earning its
   complexity?"
3. IMPORTANCE. XGBoost's default gain importance (the metric the leakage cap uses) next to
   total gain, mean |SHAP| and permutation importance. They rank features differently.
4. UNCERTAINTY. A customer-level (cluster) bootstrap of the test set, and a sweep over
   generator seeds, so every headline number can be quoted with its spread.

Training-based numbers here move slightly across operating systems and CPUs (XGBoost's
histogram arithmetic is not bit-identical everywhere); the conclusions do not.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from typing import Any

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config import COST_FN, COST_FP, FEATURES, RANDOM_SEED
from data.synthetic import generate_dataset
from evaluate.metrics import classification_report_at
from models.train import (
    CALIBRATOR_PATH,
    METADATA_PATH,
    MODEL_PATH,
    build_encoder,
    fit_calibrator,
    fit_model,
    pick_threshold,
    prepare_features,
    split_by_customer,
)

logger = logging.getLogger(__name__)

NUMERIC = [f for f in FEATURES if f != "category_encoded"]


def _scores(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    return {
        "pr_auc": round(float(average_precision_score(y, p)), 4),
        "roc_auc": round(float(roc_auc_score(y, p)), 4),
        "brier": round(float(brier_score_loss(y, p)), 5),
    }


def _oracle(frame: pd.DataFrame) -> np.ndarray:
    """P(fraud | latent traits) -- the exact label probability used by the generator."""
    return np.minimum(
        0.55 * frame["_is_fraudster"].astype(float)
        + 0.08 * frame["_is_burner"].astype(float)
        + 0.04,
        0.95,
    ).to_numpy()


def _load_or_train(X_tr, y_tr, X_va, y_va):
    """Use the committed artefacts when present, so the diagnostics describe the model
    the README and model card describe; otherwise train one the same way run.py does."""
    if all(os.path.exists(p) for p in (MODEL_PATH, CALIBRATOR_PATH, METADATA_PATH)):
        with open(METADATA_PATH, encoding="utf-8") as f:
            meta = json.load(f)
        return joblib.load(MODEL_PATH), joblib.load(CALIBRATOR_PATH), float(meta["threshold"]), "committed"
    model = fit_model(X_tr, y_tr, X_va, y_va)
    cal = fit_calibrator(model.predict_proba(X_va)[:, 1], y_va)
    thr, _ = pick_threshold(cal.predict(model.predict_proba(X_va)[:, 1]), y_va)
    return model, cal, thr, "trained_now"


def main(n_customers: int = 60_000, n_seeds: int = 5, n_boot: int = 500) -> dict[str, Any]:
    out: dict[str, Any] = {"n_customers": n_customers}

    df = generate_dataset(n_customers=n_customers, rng=np.random.default_rng(RANDOM_SEED),
                          keep_latent=True)
    enc = build_encoder()
    tr, va, te = split_by_customer(df, seed=RANDOM_SEED)
    X_tr, X_va, X_te = (prepare_features(d, enc) for d in (tr, va, te))
    y_tr, y_va, y_te = (d["is_fraud"].to_numpy() for d in (tr, va, te))

    model, cal, thr, source = _load_or_train(X_tr, y_tr, X_va, y_va)
    p_te = cal.predict(model.predict_proba(X_te)[:, 1])
    out["model_source"] = source
    out["threshold"] = round(thr, 4)

    # --- 1. Ceiling ------------------------------------------------------------------
    oracle = _oracle(te)
    model_scores = _scores(y_te, p_te)
    oracle_scores = _scores(y_te, oracle)
    bayes_t = COST_FP / (COST_FP + COST_FN)
    oracle_op = classification_report_at(y_te, oracle, bayes_t)
    model_op = classification_report_at(y_te, p_te, thr)
    approve_all = model_op["approve_all_cost_per_order"]
    out["ceiling"] = {
        "oracle": oracle_scores,
        "model": model_scores,
        "model_share_of_oracle_pr_auc": round(model_scores["pr_auc"] / oracle_scores["pr_auc"], 3),
        "fraud_labels_from_non_fraudster_customers": round(
            float((df.loc[df["is_fraud"] == 1, "_is_fraudster"] == 0).mean()), 3),
        "oracle_cost_per_return_at_bayes_threshold": oracle_op["cost_per_order"],
        "model_cost_per_return": model_op["cost_per_order"],
        "approve_all_cost_per_return": approve_all,
        "model_share_of_achievable_saving": round(
            (approve_all - model_op["cost_per_order"])
            / (approve_all - oracle_op["cost_per_order"]), 3),
        "bayes_threshold_c_fp_over_c_fp_plus_c_fn": round(bayes_t, 4),
    }

    # --- 2. Baselines (same split, same features) --------------------------------------
    base: dict[str, Any] = {"project_model": model_scores}
    unweighted = xgb.XGBClassifier(**{**model.get_params(), "scale_pos_weight": 1.0})
    unweighted.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    base["xgboost_unweighted_raw"] = _scores(y_te, unweighted.predict_proba(X_te)[:, 1])

    cols = NUMERIC + ["category"]
    lr = make_pipeline(
        ColumnTransformer([("num", StandardScaler(), NUMERIC),
                           ("cat", OneHotEncoder(handle_unknown="ignore"), ["category"])]),
        LogisticRegression(max_iter=2000),
    )
    lr.fit(tr[cols], y_tr)
    base["logistic_regression"] = _scores(y_te, lr.predict_proba(te[cols])[:, 1])

    rf = RandomForestClassifier(n_estimators=300, min_samples_leaf=20, n_jobs=-1,
                                random_state=RANDOM_SEED)
    rf.fit(X_tr, y_tr)
    base["random_forest"] = _scores(y_te, rf.predict_proba(X_te)[:, 1])

    best_single = max(
        ((f, float(average_precision_score(
            y_te, -te[f] if f == "customer_account_age_days" else te[f]))) for f in NUMERIC),
        key=lambda kv: kv[1],
    )
    base["best_single_feature"] = {"feature": best_single[0], "pr_auc": round(best_single[1], 4)}
    base["no_skill_base_rate"] = round(float(y_te.mean()), 4)
    out["baselines"] = base

    rules = ((te["customer_return_rate_lt"] >= 0.30)
             | (te["same_address_returns_7d"] >= 1)
             | (te["customer_account_age_days"] < 30)).astype(float).to_numpy()
    r = classification_report_at(y_te, rules, 0.5)
    out["rules_policy"] = {
        "rule": "lifetime return rate >= 30% OR any same-address return in 7d OR account < 30 days",
        **{k: r[k] for k in ("precision", "recall", "flag_rate", "cost_per_order",
                             "cost_reduction_vs_approve_all")},
        "model_at_operating_point": {k: model_op[k] for k in (
            "precision", "recall", "flag_rate", "cost_per_order", "cost_reduction_vs_approve_all")},
    }

    # --- 3. Importance, four ways ------------------------------------------------------
    import shap

    booster = model.get_booster()

    def share(kind: str) -> pd.Series:
        s = pd.Series(booster.get_score(importance_type=kind)).reindex(FEATURES).fillna(0.0)
        return s / s.sum()

    shap_vals = np.asarray(shap.TreeExplainer(model).shap_values(X_te))
    if shap_vals.ndim == 3:
        shap_vals = shap_vals[..., -1]
    mean_abs = np.abs(shap_vals).mean(axis=0)
    perm = permutation_importance(model, X_te, y_te, scoring="average_precision",
                                  n_repeats=5, random_state=0, n_jobs=1)
    imp = pd.DataFrame({
        "gain_share": share("gain"),
        "total_gain_share": share("total_gain"),
        "mean_abs_shap_share": pd.Series(mean_abs / mean_abs.sum(), index=FEATURES),
        "permutation_pr_auc_drop": pd.Series(perm.importances_mean, index=FEATURES),
    }).sort_values("mean_abs_shap_share", ascending=False)
    out["importance"] = {f: {k: round(float(v), 4) for k, v in row.items()}
                         for f, row in imp.iterrows()}

    # --- 4a. Cluster bootstrap on the test set ------------------------------------------
    rng = np.random.default_rng(0)
    groups = te.groupby("customer_id").indices
    keys = np.array(list(groups))
    draws = []
    for _ in range(n_boot):
        idx = np.concatenate([groups[k] for k in rng.choice(keys, size=len(keys))])
        rep = classification_report_at(y_te[idx], p_te[idx], thr)
        draws.append([average_precision_score(y_te[idx], p_te[idx]), rep["precision"],
                      rep["recall"], rep["cost_per_order"], rep["cost_reduction_vs_approve_all"]])
    draws = np.array(draws)
    out["bootstrap_95ci"] = {
        name: [round(float(v), 4) for v in np.percentile(draws[:, i], [2.5, 97.5])]
        for i, name in enumerate(["pr_auc", "precision", "recall", "cost_per_return",
                                  "cost_reduction_vs_approve_all"])
    }
    out["bootstrap_95ci"]["n_resamples"] = n_boot
    out["bootstrap_95ci"]["resampling_unit"] = "customer"

    # --- 4b. Generator-seed sweep (fresh data, split and model per seed) ----------------
    rows = []
    for seed in [RANDOM_SEED] + list(range(1, n_seeds)):
        d = generate_dataset(n_customers=n_customers, rng=np.random.default_rng(seed))
        a, b, c = split_by_customer(d, seed=seed)
        Xa, Xb, Xc = (prepare_features(x, enc) for x in (a, b, c))
        ya, yb, yc = (x["is_fraud"].to_numpy() for x in (a, b, c))
        m = fit_model(Xa, ya, Xb, yb, seed=seed)
        cl = fit_calibrator(m.predict_proba(Xb)[:, 1], yb)
        t, _ = pick_threshold(cl.predict(m.predict_proba(Xb)[:, 1]), yb)
        pc = cl.predict(m.predict_proba(Xc)[:, 1])
        rp = classification_report_at(yc, pc, t)
        rows.append({"seed": seed, "pr_auc": float(average_precision_score(yc, pc)),
                     "base_rate": float(yc.mean()), "threshold": t,
                     "precision": rp["precision"], "recall": rp["recall"],
                     "cost_reduction": rp["cost_reduction_vs_approve_all"]})
    sweep = pd.DataFrame(rows)
    out["seed_sweep"] = {
        "per_seed": sweep.round(4).to_dict("records"),
        "mean": sweep.drop(columns="seed").mean().round(4).to_dict(),
        "sd": sweep.drop(columns="seed").std().round(4).to_dict(),
    }
    return out


def _print(o: dict[str, Any]) -> None:
    c, b, s = o["ceiling"], o["baselines"], o["seed_sweep"]
    print(f"\nModel source: {o['model_source']}  (threshold {o['threshold']})")
    print(f"Ceiling  : oracle PR-AUC {c['oracle']['pr_auc']:.4f}; model {c['model']['pr_auc']:.4f} "
          f"= {c['model_share_of_oracle_pr_auc']:.0%} of it")
    print(f"           {c['fraud_labels_from_non_fraudster_customers']:.0%} of fraud labels come "
          "from customers without the fraudster trait (irreducible noise)")
    print(f"           model captures {c['model_share_of_achievable_saving']:.0%} of the "
          "saving an oracle could achieve")
    print("Baselines (PR-AUC):")
    for k, v in b.items():
        if isinstance(v, dict) and "pr_auc" in v:
            label = k if k != "best_single_feature" else f"best single feature ({v['feature']})"
            print(f"   {label:48s} {v['pr_auc']:.4f}")
    print(f"   {'no-skill (base rate)':48s} {b['no_skill_base_rate']:.4f}")
    rp = o["rules_policy"]
    print(f"Rules    : precision {rp['precision']:.3f} recall {rp['recall']:.3f} "
          f"cost/return INR {rp['cost_per_order']:.2f} "
          f"({rp['cost_reduction_vs_approve_all']:.0%} below approve-all)")
    print("95% CI   :", {k: v for k, v in o["bootstrap_95ci"].items() if isinstance(v, list)})
    print(f"Seeds    : PR-AUC {s['mean']['pr_auc']:.3f} +/- {s['sd']['pr_auc']:.3f}, "
          f"recall {s['mean']['recall']:.3f} +/- {s['sd']['recall']:.3f}, "
          f"cost reduction {s['mean']['cost_reduction']:.3f} +/- {s['sd']['cost_reduction']:.3f}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="Ceiling, baselines, importance and uncertainty")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--boot", type=int, default=500)
    args = ap.parse_args()
    result = main(n_seeds=args.seeds, n_boot=args.boot)
    os.makedirs("outputs", exist_ok=True)
    with open("outputs/diagnostics.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    _print(result)
    print("\nwrote outputs/diagnostics.json")
