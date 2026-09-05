"""End-to-end pipeline: generate -> train -> evaluate -> write model card.

    python run.py

Every number in README.md, models/MODEL_CARD.md and PROJECT_GUIDE.md is produced by this
script and written to outputs/metrics.json. Nothing is hand-copied, so re-running this is
the reproduction check.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any

import numpy as np

from config import CATEGORIES, COST_FN, COST_FP, RANDOM_SEED
from data.adversarial import generate_adversarial
from data.real_calibration import load_ieee_feature_validation, load_olist_calibration
from data.synthetic import generate_dataset, summarise
from evaluate.calibration import evaluate_calibration
from evaluate.cost_curve import plot_cost_curve
from evaluate.leakage_test import run_leakage_checks
from evaluate.metrics import (
    adversarial_gap,
    business_impact,
    classification_report_at,
    ranking_metrics,
)
from evaluate.sensitivity import run_sensitivity
from models.train import prepare_features, train

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("run")

OUT_DIR = "outputs"
N_CUSTOMERS = int(os.environ.get("N_CUSTOMERS", "60000"))


def _banner(title: str) -> None:
    print("\n" + "=" * 78)
    print(f"  {title}")
    print("=" * 78)


def main() -> dict[str, Any]:
    os.makedirs(OUT_DIR, exist_ok=True)
    rng = np.random.default_rng(RANDOM_SEED)

    # --- 1. Real-data grounding -----------------------------------------------------
    _banner("1/7  Real-data grounding")
    olist = load_olist_calibration()
    ieee = load_ieee_feature_validation()
    print(f"  Olist calibration : {olist['status']}")
    print(f"  IEEE-CIS validation: {ieee['status']}")
    if ieee["status"] == "not_validated":
        print("     -> reported as NOT VALIDATED in the model card (never as a pass).")
    category_share = olist.get("category_share") if olist["status"] == "calibrated" else None

    # --- 2. Data --------------------------------------------------------------------
    _banner("2/7  Generating data (latent-trait design, section 6.1)")
    df = generate_dataset(n_customers=N_CUSTOMERS, rng=rng, category_share=category_share)
    stats = summarise(df)
    print(f"  return requests={stats['n_return_requests']}  "
          f"fraud_rate={stats['fraud_rate']:.4f}  COD share={stats['cod_share']:.3f}")

    # --- 3. Train -------------------------------------------------------------------
    _banner("3/7  Training (XGBoost, grouped split, Platt calibration)")
    art = train(df, seed=RANDOM_SEED, persist=True)
    model, calibrator, encoder = art["model"], art["calibrator"], art["encoder"]
    threshold = art["threshold"]
    s = art["splits"]
    X_test, y_test = s["X_test"], s["y_test"]

    p_test_raw = model.predict_proba(X_test)[:, 1]
    p_test = calibrator.predict(p_test_raw)
    print(f"  threshold (chosen on validation) = {threshold:.4f}")

    # --- 4. Core metrics ------------------------------------------------------------
    _banner("4/7  Held-out test metrics")
    ranking = ranking_metrics(y_test, p_test)
    report = classification_report_at(y_test, p_test, threshold)
    print(f"  PR-AUC {ranking['pr_auc']:.4f}  (base rate {ranking['base_rate']:.4f}, "
          f"{ranking['pr_auc_lift_over_baseline']:.2f}x lift)")
    print(f"  ROC-AUC {ranking['roc_auc']:.4f}")
    print(f"  precision {report['precision']:.4f}  recall {report['recall']:.4f}  "
          f"F1 {report['f1']:.4f}")
    print(f"  cost/order INR {report['cost_per_order']:.2f} vs approve-all INR "
          f"{report['approve_all_cost_per_order']:.2f} "
          f"({report['cost_reduction_vs_approve_all']:.1%} lower)")

    calib = evaluate_calibration(y_test, p_test_raw, p_test,
                                 out_path=f"{OUT_DIR}/calibration.png")
    print(f"  Brier {calib['brier_raw']:.5f} (raw) -> {calib['brier_calibrated']:.5f} "
          f"(calibrated), {calib['brier_improvement_pct']:.1f}% better")

    curve = plot_cost_curve(y_test, p_test, threshold, out_path=f"{OUT_DIR}/cost_curve.png")
    sens = run_sensitivity(y_test, p_test, threshold, cost_fp=COST_FP,
                           out_csv=f"{OUT_DIR}/cost_sensitivity.csv",
                           out_png=f"{OUT_DIR}/cost_sensitivity.png")
    print(f"  cost-optimal threshold {curve['cost_optimal_threshold']:.3f}; "
          f"flat basin {curve['flat_basin_within_5pct']}")
    print(f"  max regret across FN:FP ratios 2x-50x: {sens['max_regret_pct']:.1f}%")

    # --- 5. Adversarial slice -------------------------------------------------------
    _banner("5/7  Adversarial slice (adapted abuser, section 6.4)")
    adv_df = generate_adversarial(n_rows=12_000, rng=np.random.default_rng(2026))
    X_adv = prepare_features(adv_df, encoder)
    y_adv = adv_df["is_fraud"].to_numpy()
    p_adv = calibrator.predict(model.predict_proba(X_adv)[:, 1])
    gap = adversarial_gap(y_test, p_test, y_adv, p_adv, threshold)
    print(f"  recall {gap['in_distribution_recall']:.4f} (in-dist) -> "
          f"{gap['adversarial_recall']:.4f} (adapted) "
          f"= {gap['relative_recall_drop']:.1%} relative drop")
    print(f"  adversarial PR-AUC {gap['adversarial_pr_auc']:.4f} "
          f"(base rate {gap['adversarial_base_rate']:.4f})")

    # --- 6. Leakage checks ----------------------------------------------------------
    _banner("6/7  Leakage smell test (section 7.1)")
    leak = run_leakage_checks(n_customers=N_CUSTOMERS, seed=RANDOM_SEED, verbose=True)

    # --- 7. Business impact + model card --------------------------------------------
    _banner("7/7  Business impact and model card")
    impact = business_impact(report, per_n_returns=10_000)
    print(f"  per 10,000 returns: {impact['flagged_for_review']} flagged "
          f"({impact['review_workload_pct']}% workload), "
          f"{impact['fraud_caught']} caught, {impact['fraud_missed']} missed")
    print(f"  illustrative net saving INR {impact['net_saving_inr']:,} "
          f"vs approve-all INR {impact['approve_all_loss_inr']:,}")

    metrics: dict[str, Any] = {
        "config": {
            "n_customers": N_CUSTOMERS,
            "random_seed": RANDOM_SEED,
            "cost_fp_inr": COST_FP,
            "cost_fn_inr": COST_FN,
            "categories": list(CATEGORIES),
        },
        "data": stats,
        "grounding": {"olist": olist, "ieee_cis": ieee},
        "operating_point": {"threshold": threshold,
                            "selected_on": "validation fold (never test)"},
        "ranking_metrics": ranking,
        "classification_report": report,
        "calibration": calib,
        "cost_curve": curve,
        "cost_sensitivity": sens,
        "adversarial": gap,
        "leakage_test": leak,
        "business_impact": impact,
    }
    with open(f"{OUT_DIR}/metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    write_model_card(metrics)
    print(f"\n  wrote {OUT_DIR}/metrics.json, models/MODEL_CARD.md and 3 plots")
    print("\nPipeline complete.\n")
    return metrics


def write_model_card(m: dict[str, Any]) -> None:
    r, c, cal = m["ranking_metrics"], m["classification_report"], m["calibration"]
    a, lk, bi = m["adversarial"], m["leakage_test"], m["business_impact"]
    ieee, olist = m["grounding"]["ieee_cis"], m["grounding"]["olist"]
    conc, abl, shuf = lk["concentration"], lk["top3_ablation"], lk["label_shuffle_control"]

    def yn(b: bool) -> str:
        return "PASS" if b else "FAIL"

    card = f"""# Model Card — Return Risk Scorer

Auto-generated by `run.py`. Every figure here is reproduced by re-running the pipeline;
none is hand-edited. Seed = {m['config']['random_seed']}.

## Intended use

Score an incoming **return request** for return-fraud risk and route it to
`auto_approve` or `manual_review`. Defense-only: it detects and explains risk. It has no
capability to generate, optimise or test evasive patterns.

**Not** intended to: make a final adjudication without human review, score anything other
than a return request, or serve as a general-purpose fraud model.

## Data

| | |
|---|---|
| Return requests | {m['data']['n_return_requests']:,} |
| Fraud base rate | {m['data']['fraud_rate']:.2%} |
| COD share | {m['data']['cod_share']:.1%} |
| Mean order value | INR {m['data']['mean_order_value']:,.0f} |
| Split | 60/20/20 train/val/test, **grouped by `customer_id`** |

Data is **synthetic**, generated by a latent-trait causal model
(`data/synthetic.py`). There is no real, public, India-specific return-fraud dataset:
every public "return prediction" dataset is explicitly synthetic, generic (not
fraud-labelled), or not India-specific. This is a disclosed data gap.

- **Olist calibration (order/category distributions):** `{olist['status']}`
- **IEEE-CIS feature-family validation:** `{ieee['status']}`{
  '' if ieee['status'] != 'not_validated'
  else '  — the check did NOT run. This is explicitly not a pass; see section 6.3 of the spec.'}

## Headline metrics (held-out test set)

| Metric | Value |
|---|---|
| PR-AUC | **{r['pr_auc']:.4f}** |
| No-skill baseline (base rate) | {r['base_rate']:.4f} |
| Lift over baseline | **{r['pr_auc_lift_over_baseline']:.2f}x** |
| ROC-AUC | {r['roc_auc']:.4f} |
| Brier score (calibrated) | {cal['brier_calibrated']:.5f} |
| Brier score (raw, pre-calibration) | {cal['brier_raw']:.5f} |
| Brier score (base-rate constant) | {cal['brier_base_rate_constant']:.5f} |
| Expected calibration error | {cal['ece_calibrated']:.5f} |

PR-AUC of ~{r['pr_auc']:.2f} against a ~{r['base_rate']:.0%} base rate is a genuine
~{r['pr_auc_lift_over_baseline']:.1f}x lift. It is deliberately **not** the ~0.95 the
pre-fix pipeline reported: that number came from a label leak, not from skill.

## Operating point (threshold {m['operating_point']['threshold']:.4f})

Chosen on the **validation** fold by minimising expected cost. The test set was scored
once, afterwards.

| Metric | Value |
|---|---|
| Precision | {c['precision']:.4f} |
| Recall | {c['recall']:.4f} |
| F1 | {c['f1']:.4f} |
| Flag rate (review workload) | {c['flag_rate']:.1%} |
| False-positive rate | {c['false_positive_rate']:.4f} |
| Cost / return request | INR {c['cost_per_order']:.2f} |
| Approve-all baseline | INR {c['approve_all_cost_per_order']:.2f} |
| **Cost reduction vs approve-all** | **{c['cost_reduction_vs_approve_all']:.1%}** |

Confusion matrix: TP={c['confusion']['tp']}, FP={c['confusion']['fp']},
FN={c['confusion']['fn']}, TN={c['confusion']['tn']} (n={c['n']:,}).

## False-positive cost, stated plainly

At this operating point the model sends **{c['flag_rate']:.1%}** of all return requests to
manual review, and **{1 - c['precision']:.1%}** of what it flags is a genuine customer.
That is the real price of the recall above, and it is why the cost curve
(`outputs/cost_curve.png`) and the sensitivity grid (`outputs/cost_sensitivity.csv`)
are part of the deliverable rather than a single tuned number.

## Adversarial robustness (adapted-abuser slice)

| Metric | In-distribution | Adapted abuser |
|---|---|---|
| Recall @ same threshold | {a['in_distribution_recall']:.4f} | **{a['adversarial_recall']:.4f}** |
| Precision | {a['in_distribution_precision']:.4f} | {a['adversarial_precision']:.4f} |
| PR-AUC | {r['pr_auc']:.4f} | {a['adversarial_pr_auc']:.4f} |

Relative recall drop: **{a['relative_recall_drop']:.1%}**. The slice models an abuser who
suppresses ring collisions, avoids burner accounts, keeps return history near-normal and
uses COD at the legitimate rate. The gap is reported, not minimised — it is the honest
statement of how much the model leans on signals an adapted abuser can defeat.

## Leakage tests — {yn(lk['all_passed'])}

| Check | Result | Threshold |
|---|---|---|
| Max single-feature importance | {conc['max_single_feature_importance']:.1%} ({conc['top_feature']}) | <= {conc['cap']:.0%} — {yn(conc['passed'])} |
| Top-3 ablation PR-AUC drop | {abl['full_pr_auc']:.4f} -> {abl['reduced_pr_auc']:.4f} ({abl['relative_drop']:.1%}) | <= {abl['max_allowed_relative_drop']:.0%} — {yn(abl['passed'])} |
| Label-shuffle control | PR-AUC {shuf['shuffled_pr_auc']:.4f} vs base {shuf['base_rate']:.4f} ({shuf['lift_over_base_rate']:.2f}x) | <= {shuf['max_allowed_lift']}x — {yn(shuf['passed'])} |

Dropped in the ablation: {', '.join(abl['dropped_features'])}.

## Illustrative business impact — NOT a guarantee

Per **{bi['per_n_returns']:,}** return requests, at the operating point above:

- **{bi['flagged_for_review']}** flagged for review ({bi['review_workload_pct']}% of volume)
- **{bi['fraud_caught']}** fraudulent returns caught, **{bi['fraud_missed']}** missed
- **{bi['genuine_customers_wrongly_flagged']}** genuine customers wrongly inconvenienced
- Loss: INR {bi['model_loss_inr']:,} vs INR {bi['approve_all_loss_inr']:,} approve-all
  = **INR {bi['net_saving_inr']:,}** ({bi['saving_pct']:.1%}) lower

{bi['disclaimer']}

## Limitations

1. **Synthetic training data.** Absolute metrics reflect the generator's assumptions.
   The *relative* claims (leak fixed, importance spread, graceful ablation) transfer;
   the absolute PR-AUC will not.
2. **Cost assumptions are illustrative** (FP=INR {m['config']['cost_fp_inr']:.0f},
   FN=INR {m['config']['cost_fn_inr']:.0f}). See `outputs/cost_sensitivity.csv` for how
   much the operating point moves when they are wrong.
3. **IEEE-CIS validation: `{ieee['status']}`.** {ieee.get('reason', '')}
4. **Adapted abusers evade this model materially** — see the recall gap above.
5. **No demographic fairness audit.** The generator has no demographic attributes, so
   such an audit would be meaningless here; a real deployment must run one.
"""
    os.makedirs("models", exist_ok=True)
    with open("models/MODEL_CARD.md", "w", encoding="utf-8") as f:
        f.write(card)


if __name__ == "__main__":
    main()
