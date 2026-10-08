"""Core metrics, cost accounting and business-impact translation.

Every number the model card reports comes from here, so that nothing in the docs can
drift from what the pipeline actually computes.

WHAT "COST" MEANS
-----------------
Expected INR lost per return request processed:

    cost_per_order = (FP * cost_fp + FN * cost_fn) / n

The comparison baseline is APPROVE-ALL -- what the merchant loses today with no model at
all: every fraudulent return is approved (n * base_rate * cost_fn) and nobody legitimate
is inconvenienced (no FP cost). That is the honest baseline, because a merchant's
alternative to this system is not a random classifier, it is doing nothing. We also
report REVIEW-ALL for completeness, since some merchants do manually review every return.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    roc_auc_score,
)

from config import COST_FN, COST_FP


def confusion_at(y_true: np.ndarray, p: np.ndarray, threshold: float) -> dict[str, int]:
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def cost_of(cm: dict[str, int], n: int, cost_fp: float = COST_FP,
            cost_fn: float = COST_FN) -> float:
    return (cm["fp"] * cost_fp + cm["fn"] * cost_fn) / max(n, 1)


def classification_report_at(
    y_true: np.ndarray,
    p: np.ndarray,
    threshold: float,
    cost_fp: float = COST_FP,
    cost_fn: float = COST_FN,
) -> dict[str, Any]:
    n = len(y_true)
    cm = confusion_at(y_true, p, threshold)
    tp, fp, fn = cm["tp"], cm["fp"], cm["fn"]
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    base_rate = float(y_true.mean())

    model_cost = cost_of(cm, n, cost_fp, cost_fn)
    approve_all_cost = base_rate * cost_fn
    review_all_cost = (1 - base_rate) * cost_fp

    return {
        "threshold": round(float(threshold), 4),
        "n": n,
        "base_rate": round(base_rate, 4),
        "confusion": cm,
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1": round(float(f1), 4),
        "flag_rate": round(float((p >= threshold).mean()), 4),
        "false_positive_rate": round(fp / max(cm["tn"] + fp, 1), 4),
        "cost_per_order": round(float(model_cost), 4),
        "approve_all_cost_per_order": round(float(approve_all_cost), 4),
        "review_all_cost_per_order": round(float(review_all_cost), 4),
        "cost_reduction_vs_approve_all": round(
            float(1 - model_cost / approve_all_cost) if approve_all_cost > 0 else 0.0, 4
        ),
    }


def ranking_metrics(y_true: np.ndarray, p: np.ndarray) -> dict[str, Any]:
    base_rate = float(y_true.mean())
    pr_auc = float(average_precision_score(y_true, p))
    return {
        "pr_auc": round(pr_auc, 4),
        "roc_auc": round(float(roc_auc_score(y_true, p)), 4),
        "base_rate": round(base_rate, 4),
        "pr_auc_lift_over_baseline": round(pr_auc / base_rate if base_rate else 0.0, 2),
        "brier_score": round(float(brier_score_loss(y_true, p)), 5),
    }


def adversarial_gap(
    y_test: np.ndarray,
    p_test: np.ndarray,
    y_adv: np.ndarray,
    p_adv: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    """Recall on the in-distribution test set vs the adapted-abuser slice.

    Both measured at the SAME operating threshold -- the number a merchant cares about is
    how much fraud slips through when the abuser adapts, holding the review budget fixed.
    """
    ind = classification_report_at(y_test, p_test, threshold)
    adv = classification_report_at(y_adv, p_adv, threshold)
    r_in, r_adv = ind["recall"], adv["recall"]
    return {
        "in_distribution_recall": r_in,
        "adversarial_recall": r_adv,
        "absolute_recall_gap": round(r_in - r_adv, 4),
        "relative_recall_drop": round((r_in - r_adv) / r_in if r_in else 0.0, 4),
        "in_distribution_precision": ind["precision"],
        "adversarial_precision": adv["precision"],
        "adversarial_pr_auc": round(float(average_precision_score(y_adv, p_adv)), 4),
        "adversarial_base_rate": adv["base_rate"],
        "interpretation": (
            "Recall measured at the identical threshold on an adapted-abuser slice that "
            "suppresses ring collisions, burner accounts and COD skew. The gap is the "
            "honest cost of relying on those signals; it is reported, not minimised."
        ),
    }


def business_impact(
    report: dict[str, Any],
    per_n_returns: int = 10_000,
    cost_fp: float = COST_FP,
    cost_fn: float = COST_FN,
) -> dict[str, Any]:
    """ILLUSTRATIVE order-of-magnitude translation of the measured confusion matrix.

    Scales the measured confusion matrix up to a round volume. This is arithmetic on top
    of measured rates, not a forecast: it inherits every assumption in the cost model and
    the synthetic-data caveat. Not a guarantee of realised savings.
    """
    n = report["n"]
    scale = per_n_returns / max(n, 1)
    cm = report["confusion"]
    flagged = (cm["tp"] + cm["fp"]) * scale
    caught = cm["tp"] * scale
    missed = cm["fn"] * scale
    wrongly_flagged = cm["fp"] * scale

    model_cost = report["cost_per_order"] * per_n_returns
    approve_all = report["approve_all_cost_per_order"] * per_n_returns
    return {
        "per_n_returns": per_n_returns,
        "flagged_for_review": int(round(flagged)),
        "fraud_caught": int(round(caught)),
        "fraud_missed": int(round(missed)),
        "genuine_customers_wrongly_flagged": int(round(wrongly_flagged)),
        "review_workload_pct": round(100 * flagged / per_n_returns, 1),
        "approve_all_loss_inr": int(round(approve_all)),
        "model_loss_inr": int(round(model_cost)),
        "net_saving_inr": int(round(approve_all - model_cost)),
        "saving_pct": report["cost_reduction_vs_approve_all"],
        "disclaimer": (
            "ILLUSTRATIVE, order-of-magnitude only. Derived from measured confusion "
            f"rates on synthetic held-out data, scaled to {per_n_returns:,} returns, "
            f"using assumed FP=INR {cost_fp:.0f} / FN=INR {cost_fn:.0f}. A real merchant "
            "must substitute their own cost-of-goods, reverse-logistics and review-cost "
            "figures before treating any of this as a savings estimate."
        ),
    }
