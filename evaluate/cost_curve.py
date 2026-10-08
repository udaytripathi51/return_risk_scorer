"""Cost-vs-threshold curve.

A single threshold is a claim without evidence behind it. This module sweeps the whole
operating range and plots expected cost per return request, so a reviewer can see where
the optimum sits AND how flat the basin around it is. Flatness matters more than the
optimum: if cost barely moves between 0.08 and 0.20, the exact threshold is a business
choice about review capacity, not a number the model has to nail.
"""
from __future__ import annotations

import os
from typing import Any

import matplotlib

matplotlib.use("Agg")  # headless: CI and Docker have no display
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from config import COST_FN, COST_FP  # noqa: E402
from evaluate.metrics import confusion_at, cost_of  # noqa: E402


def sweep_thresholds(
    y_true: np.ndarray,
    p: np.ndarray,
    cost_fp: float = COST_FP,
    cost_fn: float = COST_FN,
    n_points: int = 199,
) -> pd.DataFrame:
    rows = []
    n = len(y_true)
    for t in np.linspace(0.01, 0.99, n_points):
        cm = confusion_at(y_true, p, t)
        tp, fp, fn = cm["tp"], cm["fp"], cm["fn"]
        rows.append(
            {
                "threshold": float(t),
                "cost_per_order": cost_of(cm, n, cost_fp, cost_fn),
                "precision": tp / max(tp + fp, 1),
                "recall": tp / max(tp + fn, 1),
                "flag_rate": float((p >= t).mean()),
            }
        )
    return pd.DataFrame(rows)


def plot_cost_curve(
    y_true: np.ndarray,
    p: np.ndarray,
    chosen_threshold: float,
    out_path: str = "outputs/cost_curve.png",
    cost_fp: float = COST_FP,
    cost_fn: float = COST_FN,
) -> dict[str, Any]:
    df = sweep_thresholds(y_true, p, cost_fp, cost_fn)
    base_rate = float(y_true.mean())
    approve_all = base_rate * cost_fn
    review_all = (1 - base_rate) * cost_fp

    best = df.loc[df["cost_per_order"].idxmin()]
    chosen_cost = float(
        np.interp(chosen_threshold, df["threshold"], df["cost_per_order"])
    )

    # Width of the basin within 5% of the minimum -- the "how much does the exact
    # threshold matter" answer.
    tol = best["cost_per_order"] * 1.05
    within = df.loc[df["cost_per_order"] <= tol, "threshold"]
    basin = (float(within.min()), float(within.max())) if len(within) else (np.nan, np.nan)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    ax1.plot(df["threshold"], df["cost_per_order"], lw=2, color="#1f4e79",
             label="Model cost/order")
    ax1.axhline(approve_all, ls="--", color="#c0392b",
                label=f"Approve-all baseline (INR {approve_all:.1f})")
    ax1.axhline(review_all, ls=":", color="#7f8c8d",
                label=f"Review-all baseline (INR {review_all:.1f})")
    ax1.axvline(chosen_threshold, color="#27ae60", lw=1.6,
                label=f"Chosen t={chosen_threshold:.3f} (INR {chosen_cost:.1f})")
    if not np.isnan(basin[0]):
        ax1.axvspan(basin[0], basin[1], color="#27ae60", alpha=0.10,
                    label="within 5% of optimum")
    ax1.set_xlabel("Decision threshold")
    ax1.set_ylabel("Expected cost per return request (INR)")
    ax1.set_title(f"Cost curve  (FP=INR {cost_fp:.0f}, FN=INR {cost_fn:.0f})")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)

    ax2.plot(df["threshold"], df["precision"], lw=2, label="Precision", color="#8e44ad")
    ax2.plot(df["threshold"], df["recall"], lw=2, label="Recall", color="#e67e22")
    ax2.plot(df["threshold"], df["flag_rate"], lw=1.5, ls="--",
             label="Flag rate (review workload)", color="#7f8c8d")
    ax2.axvline(chosen_threshold, color="#27ae60", lw=1.6)
    ax2.set_xlabel("Decision threshold")
    ax2.set_ylabel("Rate")
    ax2.set_title("Precision / recall / review workload")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)

    return {
        "cost_optimal_threshold": round(float(best["threshold"]), 4),
        "cost_at_optimum": round(float(best["cost_per_order"]), 4),
        "chosen_threshold": round(float(chosen_threshold), 4),
        "cost_at_chosen": round(chosen_cost, 4),
        "approve_all_cost": round(float(approve_all), 4),
        "review_all_cost": round(float(review_all), 4),
        "flat_basin_within_5pct": [round(basin[0], 4), round(basin[1], 4)],
        "plot": out_path,
    }
