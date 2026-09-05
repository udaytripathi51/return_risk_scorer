"""Probability calibration reporting (PROJECT_SPEC.md section 7.2).

The deliverable is a risk *score*, not just a binary action, so the number on the 0-1
axis has to mean something. "0.7" should mean roughly 70 out of 100 such requests are
fraudulent. Ranking metrics (PR-AUC, ROC-AUC) say nothing about this -- they are
invariant to any monotone rescaling.

WHY CALIBRATION IS NEEDED HERE SPECIFICALLY
-------------------------------------------
`scale_pos_weight` (used to handle the ~12% class imbalance) re-weights the positive
class during training. That deliberately breaks the probabilistic interpretation of the
output: raw scores come out systematically inflated. So the raw model is a good *ranker*
and a bad *probability estimator*, and the fix is a Platt calibrator fitted on the
validation fold -- STRICTLY monotone, so it introduces no ties and PR-AUC, ROC-AUC and
every SHAP attribution are preserved exactly, while the Brier score improves.
(Isotonic regression was measured first and rejected: its step function tied thousands of
rows together and cost ~3% of PR-AUC. See models/train.py.)

This module reports Brier BEFORE and AFTER, plus a reliability diagram, so the
improvement is visible rather than asserted.
"""
from __future__ import annotations

import os
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import brier_score_loss  # noqa: E402


def reliability_table(
    y_true: np.ndarray, p: np.ndarray, n_bins: int = 10
) -> list[dict[str, Any]]:
    """Predicted-probability bucket vs actual fraud rate in that bucket."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, edges, right=True) - 1, 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        m = idx == b
        if not m.any():
            continue
        rows.append(
            {
                "bin": f"[{edges[b]:.1f}, {edges[b + 1]:.1f})",
                "n": int(m.sum()),
                "mean_predicted": round(float(p[m].mean()), 4),
                "actual_fraud_rate": round(float(y_true[m].mean()), 4),
                "gap": round(float(p[m].mean() - y_true[m].mean()), 4),
            }
        )
    return rows


def expected_calibration_error(
    y_true: np.ndarray, p: np.ndarray, n_bins: int = 10
) -> float:
    """Sample-weighted mean |predicted - actual| across buckets."""
    rows = reliability_table(y_true, p, n_bins)
    n = len(y_true)
    return float(sum(r["n"] * abs(r["gap"]) for r in rows) / max(n, 1))


def evaluate_calibration(
    y_true: np.ndarray,
    p_raw: np.ndarray,
    p_calibrated: np.ndarray,
    out_path: str = "outputs/calibration.png",
    n_bins: int = 10,
) -> dict[str, Any]:
    brier_raw = float(brier_score_loss(y_true, p_raw))
    brier_cal = float(brier_score_loss(y_true, p_calibrated))
    ece_raw = expected_calibration_error(y_true, p_raw, n_bins)
    ece_cal = expected_calibration_error(y_true, p_calibrated, n_bins)

    # A useful floor: always predicting the base rate. Any model whose Brier score is not
    # below this is adding nothing over a constant.
    base_rate = float(y_true.mean())
    brier_base = float(brier_score_loss(y_true, np.full_like(p_raw, base_rate)))

    tbl_raw = reliability_table(y_true, p_raw, n_bins)
    tbl_cal = reliability_table(y_true, p_calibrated, n_bins)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

    ax1.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect calibration")
    ax1.plot([r["mean_predicted"] for r in tbl_raw],
             [r["actual_fraud_rate"] for r in tbl_raw],
             "o-", lw=2, color="#c0392b",
             label=f"Raw (scale_pos_weight)  Brier={brier_raw:.4f}")
    ax1.plot([r["mean_predicted"] for r in tbl_cal],
             [r["actual_fraud_rate"] for r in tbl_cal],
             "s-", lw=2, color="#1f4e79",
             label=f"Platt-calibrated  Brier={brier_cal:.4f}")
    ax1.axhline(base_rate, ls=":", color="#7f8c8d",
                label=f"Base rate {base_rate:.3f}")
    ax1.set_xlabel("Mean predicted probability")
    ax1.set_ylabel("Observed fraud rate")
    ax1.set_title("Reliability diagram (held-out test set)")
    ax1.legend(fontsize=8, loc="upper left")
    ax1.grid(alpha=0.3)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)

    ax2.hist(p_calibrated, bins=40, color="#1f4e79", alpha=0.75)
    ax2.set_yscale("log")
    ax2.set_xlabel("Calibrated risk score")
    ax2.set_ylabel("Count (log scale)")
    ax2.set_title("Distribution of served risk scores")
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)

    return {
        "brier_raw": round(brier_raw, 5),
        "brier_calibrated": round(brier_cal, 5),
        "brier_base_rate_constant": round(brier_base, 5),
        "brier_improvement_pct": round(
            100 * (brier_raw - brier_cal) / brier_raw if brier_raw > 0 else 0.0, 2
        ),
        "ece_raw": round(ece_raw, 5),
        "ece_calibrated": round(ece_cal, 5),
        "reliability_raw": tbl_raw,
        "reliability_calibrated": tbl_cal,
        "plot": out_path,
        "note": (
            "Platt calibration is fitted on the validation fold only and is strictly "
            "monotone, so PR-AUC, ROC-AUC and all SHAP attributions are identical "
            "before and after -- only the probability scale changes. Isotonic was "
            "measured first and rejected: its ties cost ~3% of PR-AUC."
        ),
    }
