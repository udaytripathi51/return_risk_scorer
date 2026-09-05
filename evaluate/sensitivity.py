"""Cost-assumption sensitivity analysis (PROJECT_SPEC.md sections 7.4 and 2).

The FP=INR 50 / FN=INR 500 figures are illustrative. The honest question is therefore not
"what is the optimal threshold" but "how much does the answer move if those numbers are
wrong?" This module re-optimises the threshold across a grid of FN:FP ratios and reports
how the operating point, recall and review workload shift.

This is what turns an asserted cost assumption into a defensible one: a reviewer can see
that being wrong by 2x on the ratio moves the threshold but not the conclusion.
"""
from __future__ import annotations

import os
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from config import COST_FN, COST_FP  # noqa: E402
from evaluate.metrics import confusion_at, cost_of  # noqa: E402

# FN:FP ratios spanning "review is nearly as expensive as the loss" (2x) to "the loss
# dwarfs review" (50x). The project's own assumption sits at 10x.
RATIOS = [2, 5, 10, 20, 50]


def _best_threshold(y: np.ndarray, p: np.ndarray, cost_fp: float,
                    cost_fn: float) -> tuple[float, dict[str, int]]:
    n = len(y)
    best_t, best_c, best_cm = 0.5, float("inf"), {}
    for t in np.linspace(0.01, 0.99, 199):
        cm = confusion_at(y, p, t)
        c = cost_of(cm, n, cost_fp, cost_fn)
        if c < best_c:
            best_t, best_c, best_cm = float(t), c, cm
    return best_t, best_cm


def run_sensitivity(
    y_true: np.ndarray,
    p: np.ndarray,
    chosen_threshold: float,
    cost_fp: float = COST_FP,
    out_csv: str = "outputs/cost_sensitivity.csv",
    out_png: str = "outputs/cost_sensitivity.png",
) -> dict[str, Any]:
    n = len(y_true)
    base_rate = float(y_true.mean())
    rows = []

    for ratio in RATIOS:
        cost_fn = cost_fp * ratio
        t_opt, cm_opt = _best_threshold(y_true, p, cost_fp, cost_fn)
        cm_chosen = confusion_at(y_true, p, chosen_threshold)

        def stats(cm: dict[str, int]) -> tuple[float, float, float, float]:
            tp, fp, fn = cm["tp"], cm["fp"], cm["fn"]
            return (
                tp / max(tp + fp, 1),
                tp / max(tp + fn, 1),
                (tp + fp) / n,
                cost_of(cm, n, cost_fp, cost_fn),
            )

        p_o, r_o, flag_o, c_o = stats(cm_opt)
        _, _, _, c_c = stats(cm_chosen)
        approve_all = base_rate * cost_fn

        rows.append(
            {
                "fn_fp_ratio": ratio,
                "cost_fp": cost_fp,
                "cost_fn": cost_fn,
                "optimal_threshold": round(t_opt, 4),
                "precision_at_optimal": round(p_o, 4),
                "recall_at_optimal": round(r_o, 4),
                "flag_rate_at_optimal": round(flag_o, 4),
                "cost_at_optimal": round(c_o, 4),
                "cost_at_project_threshold": round(c_c, 4),
                # How much worse is it to keep our chosen threshold when the true cost
                # ratio is different? This is the regret from being wrong.
                "regret_pct": round(100 * (c_c - c_o) / c_o if c_o > 0 else 0.0, 2),
                "approve_all_cost": round(approve_all, 4),
                "saving_vs_approve_all_pct": round(
                    100 * (1 - c_o / approve_all) if approve_all > 0 else 0.0, 2
                ),
            }
        )

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    df.to_csv(out_csv, index=False)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    ax1.plot(df["fn_fp_ratio"], df["optimal_threshold"], "o-", lw=2, color="#1f4e79")
    ax1.axhline(chosen_threshold, ls="--", color="#27ae60",
                label=f"Project threshold {chosen_threshold:.3f}")
    ax1.axvline(COST_FN / COST_FP, ls=":", color="#c0392b",
                label=f"Project assumption ({COST_FN / COST_FP:.0f}x)")
    ax1.set_xscale("log")
    ax1.set_xlabel("FN : FP cost ratio (log scale)")
    ax1.set_ylabel("Cost-optimal threshold")
    ax1.set_title("Optimal threshold vs cost assumption")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)

    ax2.plot(df["fn_fp_ratio"], df["recall_at_optimal"], "o-", lw=2,
             label="Recall at optimum", color="#e67e22")
    ax2.plot(df["fn_fp_ratio"], df["flag_rate_at_optimal"], "s--", lw=2,
             label="Review workload", color="#7f8c8d")
    ax2.plot(df["fn_fp_ratio"], df["regret_pct"] / 100, "^:", lw=2,
             label="Regret of keeping our threshold", color="#c0392b")
    ax2.set_xscale("log")
    ax2.set_xlabel("FN : FP cost ratio (log scale)")
    ax2.set_ylabel("Rate / fraction")
    ax2.set_title("What being wrong about cost actually does")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_png, dpi=140)
    plt.close(fig)

    return {
        "grid": rows,
        "csv": out_csv,
        "plot": out_png,
        "max_regret_pct": round(float(df["regret_pct"].max()), 2),
        "interpretation": (
            "The threshold chosen under the project's 10x assumption is re-scored "
            "against every other ratio in the grid. `regret_pct` is how much worse our "
            "fixed threshold performs than the threshold that ratio would have chosen -- "
            "the practical cost of the cost assumption being wrong."
        ),
    }
