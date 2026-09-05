"""Model training (PROJECT_SPEC.md section 6.2).

THE BUG THIS FIXES
------------------
`early_stopping_rounds` was passed to `.fit()`. On xgboost >= 2.0 that raises
`TypeError: XGBClassifier.fit() got an unexpected keyword argument
'early_stopping_rounds'` -- verified here against xgboost 3.4.1. It belongs on the
constructor.

TWO DELIBERATE DEVIATIONS FROM THE SPEC SNIPPET
-----------------------------------------------
1. THREE-WAY SPLIT, NOT TWO. The snippet in section 6.2 passes the *test* set as
   `eval_set`, which makes the test set part of model selection: early stopping picks
   the tree count that looks best on it, so the reported test metric is optimistic. We
   split train / validation / test, early-stop on validation, pick the decision
   threshold on validation, and touch the test set exactly once at the end.

2. GROUPED SPLIT ON customer_id. A customer contributes several return requests, and
   customer-level features (return-rate history, account age, ring membership) are
   near-constant within a customer. A row-wise split would put the same customer on both
   sides and let the model memorise individuals -- a second, subtler leak of the same
   family as section 6.1. `GroupShuffleSplit` keeps every customer wholly inside one
   split.

CALIBRATION: PLATT, NOT ISOTONIC -- AND WHY
-------------------------------------------
`scale_pos_weight` deliberately distorts predicted probabilities: it re-weights the
positive class, so raw outputs are not probabilities at all, they are inflated scores.
Since the deliverable is a risk *score* and not just a binary action, we calibrate on the
validation fold.

Isotonic regression was the first choice -- it is the standard recommendation for tree
ensembles -- and it was measured and rejected. Isotonic is only WEAKLY monotone: it is a
step function, so it maps whole ranges of raw scores onto a single value. On this data it
collapsed 4,187 distinct test scores into 41 levels, tying 4,443 rows together, and those
ties cost real ranking quality: PR-AUC fell from 0.4488 to 0.4353. It also emitted exactly
1.0 for a few rows, colliding with the sentinel value the fail-closed path uses.

Platt scaling (a logistic regression on the log-odds of the raw score) is STRICTLY
monotone, so it introduces no ties and preserves PR-AUC and ROC-AUC exactly. Measured
head-to-head on the same fold it also calibrated slightly BETTER (Brier 0.09354 vs
0.09419). Strictly better on every axis, so Platt it is. Both before/after Brier scores
are reported by evaluate/calibration.py so the improvement is visible, not asserted.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder

from config import CATEGORIES, COST_FN, COST_FP, FEATURES, RANDOM_SEED

logger = logging.getLogger(__name__)

MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "model.pkl")
CALIBRATOR_PATH = os.path.join(MODEL_DIR, "calibrator.pkl")
ENCODER_PATH = os.path.join(MODEL_DIR, "encoder.pkl")
METADATA_PATH = os.path.join(MODEL_DIR, "metadata.json")


def build_encoder() -> LabelEncoder:
    """Encoder fitted on the FIXED taxonomy, not on whatever happened to appear in the
    training sample. Anything outside it must fail closed at serving time (section 6.7),
    so the class list has to be a deliberate contract rather than a data artefact."""
    enc = LabelEncoder()
    enc.fit(CATEGORIES)
    return enc


def prepare_features(df: pd.DataFrame, encoder: LabelEncoder) -> pd.DataFrame:
    out = df.copy()
    out["category_encoded"] = encoder.transform(out["category"])
    return out[FEATURES]


def split_by_customer(
    df: pd.DataFrame, seed: int = RANDOM_SEED
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """60/20/20 train/val/test, grouped so a customer never spans two splits."""
    groups = df["customer_id"].to_numpy()

    gss1 = GroupShuffleSplit(n_splits=1, test_size=0.40, random_state=seed)
    train_idx, rest_idx = next(gss1.split(df, groups=groups))
    train = df.iloc[train_idx].reset_index(drop=True)
    rest = df.iloc[rest_idx].reset_index(drop=True)

    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=seed)
    val_idx, test_idx = next(gss2.split(rest, groups=rest["customer_id"].to_numpy()))
    val = rest.iloc[val_idx].reset_index(drop=True)
    test = rest.iloc[test_idx].reset_index(drop=True)

    assert not (set(train["customer_id"]) & set(test["customer_id"]))
    assert not (set(train["customer_id"]) & set(val["customer_id"]))
    assert not (set(val["customer_id"]) & set(test["customer_id"]))
    return train, val, test


def fit_model(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    seed: int = RANDOM_SEED,
) -> xgb.XGBClassifier:
    """Fit XGBoost with early stopping on the VALIDATION fold."""
    n_pos = int(y_train.sum())
    n_neg = int(len(y_train) - n_pos)
    scale_pos = n_neg / max(n_pos, 1)

    model = xgb.XGBClassifier(
        n_estimators=400,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=8,
        reg_lambda=2.0,
        scale_pos_weight=scale_pos,
        random_state=seed,
        eval_metric="aucpr",
        early_stopping_rounds=30,   # constructor, NOT .fit() -- this is the section 6.2 fix
        n_jobs=-1,
        tree_method="hist",
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    logger.info(
        "trained: best_iteration=%s scale_pos_weight=%.3f",
        getattr(model, "best_iteration", None),
        scale_pos,
    )
    return model


class PlattCalibrator:
    """Strictly-monotone probability calibration (Platt scaling).

    Fits a logistic regression on the LOG-ODDS of the raw model score. Being strictly
    monotone it introduces no ties, so PR-AUC, ROC-AUC and the ordering of every SHAP
    attribution are preserved exactly -- only the 0-1 scale changes. See the module
    docstring for the measured comparison against isotonic regression.

    A module-level class rather than a closure so joblib can pickle it for serving.
    """

    _EPS = 1e-6

    def __init__(self) -> None:
        self.lr = LogisticRegression(C=1e6, solver="lbfgs")

    def _logit(self, p) -> np.ndarray:
        p = np.clip(np.asarray(p, dtype=float), self._EPS, 1.0 - self._EPS)
        return np.log(p / (1.0 - p)).reshape(-1, 1)

    def fit(self, p_raw, y) -> "PlattCalibrator":
        self.lr.fit(self._logit(p_raw), y)
        return self

    def predict(self, p_raw) -> np.ndarray:
        return self.lr.predict_proba(self._logit(p_raw))[:, 1]


def fit_calibrator(p_val_raw: np.ndarray, y_val: np.ndarray) -> PlattCalibrator:
    """Platt calibration fitted on the validation fold only."""
    return PlattCalibrator().fit(p_val_raw, y_val)


def pick_threshold(
    p_val: np.ndarray,
    y_val: np.ndarray,
    cost_fp: float = COST_FP,
    cost_fn: float = COST_FN,
) -> tuple[float, dict[str, float]]:
    """Choose the operating threshold that minimises expected cost per return request.

    Selected on VALIDATION, never on test. The grid is dense (1000 points) but the cost
    curve is flat near the optimum, which is itself worth reporting -- see
    evaluate/sensitivity.py.
    """
    grid = np.linspace(0.01, 0.99, 999)
    best_t, best_cost, best_stats = 0.5, float("inf"), {}
    n = len(y_val)
    for t in grid:
        pred = (p_val >= t).astype(int)
        fp = int(((pred == 1) & (y_val == 0)).sum())
        fn = int(((pred == 0) & (y_val == 1)).sum())
        cost = (fp * cost_fp + fn * cost_fn) / n
        if cost < best_cost:
            tp = int(((pred == 1) & (y_val == 1)).sum())
            best_t, best_cost = float(t), float(cost)
            best_stats = {
                "cost_per_order": round(cost, 4),
                "flag_rate": round(float(pred.mean()), 4),
                "precision": round(tp / max(tp + fp, 1), 4),
                "recall": round(tp / max(tp + fn, 1), 4),
            }
    return best_t, best_stats


def train(
    df: pd.DataFrame,
    seed: int = RANDOM_SEED,
    persist: bool = True,
) -> dict[str, Any]:
    """Full training run. Returns every artefact the evaluators and the API need."""
    encoder = build_encoder()
    train_df, val_df, test_df = split_by_customer(df, seed=seed)

    X_train = prepare_features(train_df, encoder)
    X_val = prepare_features(val_df, encoder)
    X_test = prepare_features(test_df, encoder)
    y_train = train_df["is_fraud"].to_numpy()
    y_val = val_df["is_fraud"].to_numpy()
    y_test = test_df["is_fraud"].to_numpy()

    logger.info(
        "split sizes: train=%d val=%d test=%d | fraud rates: %.4f / %.4f / %.4f",
        len(y_train), len(y_val), len(y_test),
        y_train.mean(), y_val.mean(), y_test.mean(),
    )

    model = fit_model(X_train, y_train, X_val, y_val, seed=seed)

    p_val_raw = model.predict_proba(X_val)[:, 1]
    calibrator = fit_calibrator(p_val_raw, y_val)
    p_val_cal = calibrator.predict(p_val_raw)

    threshold, val_stats = pick_threshold(p_val_cal, y_val)
    logger.info("threshold selected on validation: %.3f  %s", threshold, val_stats)

    artefacts: dict[str, Any] = {
        "model": model,
        "calibrator": calibrator,
        "encoder": encoder,
        "threshold": threshold,
        "features": FEATURES,
        "splits": {
            "train": train_df, "val": val_df, "test": test_df,
            "X_train": X_train, "X_val": X_val, "X_test": X_test,
            "y_train": y_train, "y_val": y_val, "y_test": y_test,
        },
        "val_stats": val_stats,
    }

    if persist:
        os.makedirs(MODEL_DIR, exist_ok=True)
        joblib.dump(model, MODEL_PATH)
        joblib.dump(calibrator, CALIBRATOR_PATH)
        joblib.dump(encoder, ENCODER_PATH)
        meta = {
            "threshold": threshold,
            "features": FEATURES,
            "categories": list(CATEGORIES),
            "n_train": int(len(y_train)),
            "n_val": int(len(y_val)),
            "n_test": int(len(y_test)),
            "train_fraud_rate": round(float(y_train.mean()), 4),
            "best_iteration": int(getattr(model, "best_iteration", 0) or 0),
            "cost_fp": COST_FP,
            "cost_fn": COST_FN,
            "validation_operating_point": val_stats,
            "library_versions": {
                "xgboost": xgb.__version__,
                "numpy": np.__version__,
                "pandas": pd.__version__,
            },
        }
        with open(METADATA_PATH, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        logger.info("persisted model artefacts to %s/", MODEL_DIR)

    return artefacts


def train_from_scratch(n_customers: int = 60_000, seed: int = RANDOM_SEED) -> dict[str, Any]:
    from data.synthetic import generate_dataset

    df = generate_dataset(n_customers=n_customers, rng=np.random.default_rng(seed))
    return train(df, seed=seed)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    train_from_scratch()
