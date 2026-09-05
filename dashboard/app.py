"""Reviewer-facing Streamlit dashboard (PROJECT_SPEC.md section 7.6).

    streamlit run dashboard/app.py

Three tabs:
  1. Score one order   -- manual entry, risk score, SHAP waterfall, cost-curve position
  2. Score a CSV batch -- upload, triage table, downloadable results
  3. Model evidence    -- the honest metrics: leakage checks, calibration, adversarial gap

This is what the pitch video should show: a live decision with its reasons, not a curl
command.

DEFENSE-ONLY NOTE: this dashboard runs against the same service as the API and exposes no
extra capability. It shows the operating threshold because a merchant operator needs it
to read their own triage queue -- the same per-order carve-out the API relies on. It has
no batch-probe, no threshold sweep against live scoring, and no counterfactual tool.
"""
from __future__ import annotations

import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.service import RiskScoringService  # noqa: E402
from config import CATEGORIES, CATEGORY_RETURN_BASE_RATE, COST_FN, COST_FP  # noqa: E402

st.set_page_config(page_title="Return Risk Scorer", page_icon="🛡️", layout="wide")

METRICS_PATH = "outputs/metrics.json"


@st.cache_resource
def get_service() -> RiskScoringService:
    return RiskScoringService()


@st.cache_data
def get_metrics() -> dict | None:
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH, encoding="utf-8") as f:
            return json.load(f)
    return None


service = get_service()
metrics = get_metrics()

st.title("🛡️ Return Risk Scorer")
st.caption(
    "Defense-only return-fraud risk scoring for Indian e-commerce · "
    "Razorpay AI Buildathon 2026 · Track 02: AI Risk Manager"
)

if not service.is_ready():
    st.error(
        "Model artefacts are not loaded, so every request fails closed to "
        "`manual_review`. Run `python run.py` first to train and persist the model."
    )
    st.stop()

# --- sidebar -------------------------------------------------------------------------
with st.sidebar:
    st.header("Model")
    st.metric("Version", service.model_version)
    st.metric("Operating threshold", f"{service.threshold:.4f}")
    st.metric("Features", len(service.features))
    if metrics:
        r = metrics["ranking_metrics"]
        st.metric("PR-AUC (held-out)", f"{r['pr_auc']:.4f}",
                  delta=f"{r['pr_auc_lift_over_baseline']:.2f}x vs base rate")
    st.divider()
    st.caption(
        f"Cost assumptions (illustrative): a missed fraud costs **INR {COST_FN:.0f}**, "
        f"a wrongly-flagged genuine return costs **INR {COST_FP:.0f}**."
    )

tab1, tab2, tab3 = st.tabs(
    ["Score one order", "Score a CSV batch", "Model evidence"]
)

# =====================================================================================
# TAB 1 — single order
# =====================================================================================
with tab1:
    left, right = st.columns([1, 1.35])

    with left:
        st.subheader("Return request")
        preset = st.selectbox(
            "Start from",
            ["High-risk example", "Clean example", "Borderline example"],
        )
        presets = {
            "High-risk example": dict(
                rr_lt=0.62, rr_90=0.75, age=18, vel=5, val=4899.0, disc=50.0,
                cat="Apparel", cod=1, days=13.0, addr=3, email=2, pay=1, ip=1, rto=0.44,
            ),
            "Clean example": dict(
                rr_lt=0.04, rr_90=0.0, age=1400, vel=1, val=899.0, disc=0.0,
                cat="Books", cod=0, days=2.0, addr=0, email=0, pay=0, ip=0, rto=0.05,
            ),
            "Borderline example": dict(
                rr_lt=0.22, rr_90=0.30, age=180, vel=2, val=2100.0, disc=20.0,
                cat="Home", cod=1, days=8.0, addr=1, email=0, pay=0, ip=0, rto=0.20,
            ),
        }
        d = presets[preset]

        c1, c2 = st.columns(2)
        with c1:
            rr_lt = st.slider("Lifetime return rate", 0.0, 1.0, d["rr_lt"], 0.01)
            age = st.number_input("Account age (days)", 0, 5000, d["age"])
            val = st.number_input("Order value (INR)", 1.0, 50000.0, d["val"])
            cat = st.selectbox("Category", CATEGORIES, index=CATEGORIES.index(d["cat"]))
            days = st.number_input(
                "Days to return (-1 = unknown)", -1.0, 60.0, d["days"]
            )
            addr = st.number_input("Same-address returns (7d)", 0, 50, d["addr"])
            pay = st.selectbox("Payment-hash collision", [0, 1],
                               index=d["pay"], format_func=lambda x: "Yes" if x else "No")
        with c2:
            rr_90 = st.slider("90-day return rate", 0.0, 1.0, d["rr_90"], 0.01)
            vel = st.number_input("Orders in last 7 days", 0, 100, d["vel"])
            disc = st.slider("Discount %", 0.0, 100.0, d["disc"], 1.0)
            cod = st.selectbox("Payment method", [1, 0], index=0 if d["cod"] else 1,
                               format_func=lambda x: "Cash on delivery" if x else "Prepaid")
            email = st.number_input("Same-email returns (7d)", 0, 50, d["email"])
            ip = st.selectbox("IP/phone collision", [0, 1],
                              index=d["ip"], format_func=lambda x: "Yes" if x else "No")
            rto = st.slider("Pincode RTO risk", 0.0, 1.0, d["rto"], 0.01)

        order = {
            "customer_return_rate_lt": rr_lt,
            "customer_return_rate_90d": rr_90,
            "customer_account_age_days": int(age),
            "customer_order_velocity_7d": int(vel),
            "order_value": float(val),
            "discount_percentage": float(disc),
            "category": cat,
            "category_return_base_rate": CATEGORY_RETURN_BASE_RATE[cat],
            "is_prepaid": 1 - int(cod),
            "is_cod": int(cod),
            "days_to_return": float(days),
            "same_address_returns_7d": int(addr),
            "same_email_returns_7d": int(email),
            "payment_hash_collision": int(pay),
            "ip_phone_collision": int(ip),
            "rto_risk_score": float(rto),
        }

    with right:
        st.subheader("Decision")
        result = service.score(order)
        score, action = result["risk_score"], result["action"]

        m1, m2, m3 = st.columns(3)
        m1.metric("Risk score", f"{score:.3f}")
        m2.metric("Action", "MANUAL REVIEW" if action == "manual_review" else "AUTO-APPROVE")
        m3.metric("Threshold", f"{result['threshold_used']:.3f}")

        if action == "manual_review":
            st.warning("Routed to **manual review** — a human decides this one.")
        else:
            st.success("**Auto-approved** — below the review threshold.")
        st.progress(min(score, 1.0))

        for note in result.get("data_quality_notes", []):
            st.info(note)

        st.markdown("##### Why — top 3 drivers")
        for i, reason in enumerate(result["reasons"], 1):
            arrow = "🔺" if reason.get("direction") == "increases_risk" else "🔻"
            contrib = reason.get("contribution")
            suffix = f"  ·  `{contrib:+.3f}`" if contrib is not None else ""
            st.markdown(f"{arrow} **{i}.** {reason['reason']}{suffix}")

        # --- SHAP waterfall ---------------------------------------------------------
        if service.explainer is not None and result["reasons"][0]["contribution"] is not None:
            st.markdown("##### SHAP waterfall (raw log-odds margin)")
            X, _ = service._to_frame(order)
            vals = service.explainer.shap_values(X)[0]
            base = float(np.ravel(service.explainer.explainer.expected_value)[0])

            order_idx = np.argsort(np.abs(vals))[::-1]
            top = order_idx[:8]
            rest_sum = float(vals[order_idx[8:]].sum())

            labels = [service.features[i] for i in top]
            contribs = [float(vals[i]) for i in top]
            if abs(rest_sum) > 1e-9:
                labels.append(f"{len(order_idx) - 8} other features")
                contribs.append(rest_sum)

            fig, ax = plt.subplots(figsize=(7, 4.2))
            cum = base
            for k, (lab, cv) in enumerate(zip(labels, contribs)):
                ax.barh(k, cv, left=cum, height=0.62,
                        color="#c0392b" if cv > 0 else "#1f4e79")
                ax.text(cum + cv + (0.02 if cv > 0 else -0.02), k, f"{cv:+.2f}",
                        va="center", ha="left" if cv > 0 else "right", fontsize=8)
                cum += cv
            ax.axvline(base, color="#7f8c8d", ls="--", lw=1)
            ax.text(base, len(labels) - 0.3, f" base {base:.2f}", fontsize=8,
                    color="#7f8c8d")
            ax.set_yticks(range(len(labels)))
            ax.set_yticklabels(labels, fontsize=8)
            ax.invert_yaxis()
            ax.set_xlabel("Contribution to raw log-odds margin")
            ax.grid(axis="x", alpha=0.3)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
            st.caption(
                "Red pushes risk up, blue pulls it down. These are raw-margin units, "
                "not probabilities — the Platt calibrator that maps margin to the 0-1 "
                "risk score is strictly monotone, so the ordering shown here is exactly "
                "the ordering that drives the score."
            )

    # --- cost curve position --------------------------------------------------------
    if metrics:
        st.divider()
        st.markdown("##### Where this order sits on the cost curve")
        cc = metrics["cost_curve"]
        cA, cB = st.columns([1.4, 1])
        with cA:
            if os.path.exists("outputs/cost_curve.png"):
                st.image("outputs/cost_curve.png", use_container_width=True)
        with cB:
            st.metric("This order's score", f"{score:.3f}")
            st.metric("Operating threshold", f"{cc['chosen_threshold']:.3f}")
            st.metric("Cost-optimal threshold", f"{cc['cost_optimal_threshold']:.3f}")
            lo, hi = cc["flat_basin_within_5pct"]
            st.caption(
                f"Any threshold in **[{lo:.3f}, {hi:.3f}]** lands within 5% of the "
                f"minimum cost — so the exact cut-off is a review-capacity decision, "
                f"not a number the model has to nail."
            )
            margin = score - cc["chosen_threshold"]
            st.caption(
                f"This order is **{abs(margin):.3f}** "
                f"{'above' if margin >= 0 else 'below'} the threshold."
            )

# =====================================================================================
# TAB 2 — batch
# =====================================================================================
with tab2:
    st.subheader("Score a CSV batch")
    st.caption(
        "Upload a CSV with one row per return request. Required columns are the API "
        "contract fields; `category_return_base_rate` and `is_prepaid` are derived if "
        "absent."
    )

    template = pd.DataFrame([order])
    st.download_button(
        "Download a template CSV",
        template.to_csv(index=False).encode(),
        "return_requests_template.csv",
        "text/csv",
    )

    up = st.file_uploader("CSV file", type=["csv"])
    if up is not None:
        df = pd.read_csv(up)
        if "category_return_base_rate" not in df.columns and "category" in df.columns:
            df["category_return_base_rate"] = df["category"].map(
                CATEGORY_RETURN_BASE_RATE
            ).fillna(0.13)
        if "is_prepaid" not in df.columns and "is_cod" in df.columns:
            df["is_prepaid"] = 1 - df["is_cod"]

        res = service.score_batch(df.to_dict("records"))
        summary = res["summary"]

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Scored", summary["n"])
        k2.metric("Flagged for review", summary["flagged_for_review"])
        k3.metric("Auto-approved", summary["auto_approved"])
        k4.metric("Flag rate", f"{summary['flag_rate']:.1%}")

        out = df.copy()
        out["risk_score"] = [r["risk_score"] for r in res["results"]]
        out["action"] = [r["action"] for r in res["results"]]
        out["top_reason"] = [r["reasons"][0]["reason"] for r in res["results"]]
        out = out.sort_values("risk_score", ascending=False)

        st.markdown("##### Triage queue (highest risk first)")
        st.dataframe(
            out[["risk_score", "action", "top_reason"]
                + [c for c in ("category", "order_value", "is_cod") if c in out.columns]],
            use_container_width=True,
            height=380,
        )
        st.download_button(
            "Download scored results",
            out.to_csv(index=False).encode(),
            "scored_returns.csv",
            "text/csv",
        )

        est_cost = summary["flagged_for_review"] * COST_FP
        st.caption(
            f"Reviewing {summary['flagged_for_review']} flagged requests costs roughly "
            f"INR {est_cost:,.0f} in review effort at the illustrative "
            f"INR {COST_FP:.0f}/review assumption."
        )

# =====================================================================================
# TAB 3 — evidence
# =====================================================================================
with tab3:
    if not metrics:
        st.info("Run `python run.py` to generate `outputs/metrics.json`.")
    else:
        r = metrics["ranking_metrics"]
        c = metrics["classification_report"]
        cal = metrics["calibration"]
        adv = metrics["adversarial"]
        lk = metrics["leakage_test"]
        bi = metrics["business_impact"]

        st.subheader("Held-out performance")
        k = st.columns(5)
        k[0].metric("PR-AUC", f"{r['pr_auc']:.4f}", delta=f"{r['pr_auc_lift_over_baseline']:.2f}x lift")
        k[1].metric("ROC-AUC", f"{r['roc_auc']:.4f}")
        k[2].metric("Precision", f"{c['precision']:.3f}")
        k[3].metric("Recall", f"{c['recall']:.3f}")
        k[4].metric("Brier", f"{cal['brier_calibrated']:.4f}")
        st.caption(
            f"Base rate {r['base_rate']:.1%}. PR-AUC of {r['pr_auc']:.2f} is a genuine "
            f"{r['pr_auc_lift_over_baseline']:.1f}x lift — deliberately not the ~0.95 the "
            "pre-fix pipeline reported, which came from a label leak rather than skill."
        )

        st.divider()
        st.subheader("Leakage checks — the rigor claim")
        conc, abl, shuf = lk["concentration"], lk["top3_ablation"], lk["label_shuffle_control"]
        st.dataframe(pd.DataFrame([
            {"check": "Max single-feature importance",
             "result": f"{conc['max_single_feature_importance']:.1%} ({conc['top_feature']})",
             "limit": f"<= {conc['cap']:.0%}",
             "status": "PASS" if conc["passed"] else "FAIL"},
            {"check": "Top-3 ablation PR-AUC drop",
             "result": f"{abl['full_pr_auc']:.4f} -> {abl['reduced_pr_auc']:.4f} ({abl['relative_drop']:.1%})",
             "limit": f"<= {abl['max_allowed_relative_drop']:.0%}",
             "status": "PASS" if abl["passed"] else "FAIL"},
            {"check": "Label-shuffle control",
             "result": f"PR-AUC {shuf['shuffled_pr_auc']:.4f} vs base {shuf['base_rate']:.4f}",
             "limit": f"<= {shuf['max_allowed_lift']}x base rate",
             "status": "PASS" if shuf["passed"] else "FAIL"},
        ]), use_container_width=True, hide_index=True)

        st.divider()
        cA, cB = st.columns(2)
        with cA:
            st.subheader("Calibration")
            if os.path.exists("outputs/calibration.png"):
                st.image("outputs/calibration.png", use_container_width=True)
            st.caption(
                f"Brier {cal['brier_raw']:.4f} raw -> {cal['brier_calibrated']:.4f} "
                f"calibrated ({cal['brier_improvement_pct']:.0f}% better); ECE "
                f"{cal['ece_raw']:.3f} -> {cal['ece_calibrated']:.3f}."
            )
        with cB:
            st.subheader("Adversarial slice")
            st.metric("Recall, in-distribution", f"{adv['in_distribution_recall']:.3f}")
            st.metric("Recall, adapted abuser", f"{adv['adversarial_recall']:.3f}",
                      delta=f"-{adv['relative_recall_drop']:.1%}", delta_color="inverse")
            st.caption(adv["interpretation"])

        st.divider()
        st.subheader("Cost sensitivity")
        if os.path.exists("outputs/cost_sensitivity.png"):
            st.image("outputs/cost_sensitivity.png", use_container_width=True)
        if os.path.exists("outputs/cost_sensitivity.csv"):
            st.dataframe(pd.read_csv("outputs/cost_sensitivity.csv"),
                         use_container_width=True, hide_index=True)

        st.divider()
        st.subheader("Illustrative business impact")
        b = st.columns(4)
        b[0].metric(f"Flagged / {bi['per_n_returns']:,}", f"{bi['flagged_for_review']:,}")
        b[1].metric("Fraud caught", f"{bi['fraud_caught']:,}")
        b[2].metric("Fraud missed", f"{bi['fraud_missed']:,}")
        b[3].metric("Net saving", f"INR {bi['net_saving_inr']:,}")
        st.warning(bi["disclaimer"])
