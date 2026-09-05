"""Data-generation, training and explainability tests."""
from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import average_precision_score

from config import CATEGORIES, FEATURES
from data.adversarial import generate_adversarial
from data.real_calibration import load_ieee_feature_validation, load_olist_calibration
from data.synthetic import generate_dataset
from models.explainer import RiskExplainer
from models.train import build_encoder, prepare_features, split_by_customer, train


@pytest.fixture(scope="module")
def dataset():
    return generate_dataset(n_customers=12_000, rng=np.random.default_rng(7))


@pytest.fixture(scope="module")
def trained(dataset):
    return train(dataset, seed=7, persist=False)


# --- section 6.1: the generator must not leak the label ------------------------------


def test_latent_traits_are_not_exported(dataset):
    """The latent traits drive the label. If any of them reached the dataset the model
    would read the label directly."""
    leaked = [c for c in dataset.columns if c.startswith("_")]
    assert not leaked, f"latent columns leaked into the dataset: {leaked}"
    for banned in ("_is_fraudster", "_is_burner", "_return_propensity", "_in_ring"):
        assert banned not in dataset.columns


def test_no_single_feature_separates_the_label(dataset):
    """The pre-fix generator amplified order_value / is_cod / discount / days_to_return
    for fraud rows after the label was drawn, which made each of them individually near-
    diagnostic. Every feature must now overlap heavily across classes."""
    fraud = dataset[dataset["is_fraud"] == 1]
    legit = dataset[dataset["is_fraud"] == 0]
    for col in ("order_value", "discount_percentage", "days_to_return", "is_cod"):
        # Point-biserial correlation with the label must stay modest.
        corr = abs(np.corrcoef(dataset[col].astype(float), dataset["is_fraud"])[0, 1])
        assert corr < 0.25, f"{col} correlates {corr:.3f} with the label — suspect leakage"
        # And the class-conditional means must not be far apart in SD units.
        pooled_sd = dataset[col].astype(float).std()
        if pooled_sd > 0:
            d = abs(fraud[col].mean() - legit[col].mean()) / pooled_sd
            assert d < 0.8, f"{col} has Cohen's d = {d:.2f} between classes"


def test_fraud_rate_is_realistic(dataset):
    assert 0.08 <= dataset["is_fraud"].mean() <= 0.18


def test_generator_is_deterministic_given_a_seed():
    a = generate_dataset(n_customers=3_000, rng=np.random.default_rng(11))
    b = generate_dataset(n_customers=3_000, rng=np.random.default_rng(11))
    assert a.equals(b)


def test_all_contract_features_present(dataset):
    for col in ("category", "is_prepaid", "is_cod", "rto_risk_score", "is_fraud"):
        assert col in dataset.columns
    assert set(dataset["category"]).issubset(set(CATEGORIES))
    # The contract's redundancy invariant.
    assert ((dataset["is_prepaid"] + dataset["is_cod"]) == 1).all()


# --- section 6.2 + training ----------------------------------------------------------


def test_training_completes_without_early_stopping_crash(trained):
    """Regression test for section 6.2: `early_stopping_rounds` on `.fit()` raises
    TypeError on xgboost >= 2.0. It belongs on the constructor."""
    model = trained["model"]
    assert model is not None
    assert hasattr(model, "best_iteration")
    assert model.get_params()["early_stopping_rounds"] == 30


def test_splits_share_no_customers(dataset):
    """A customer spanning two splits would let the model memorise individuals — a
    second, subtler leak."""
    tr, va, te = split_by_customer(dataset, seed=7)
    assert not (set(tr["customer_id"]) & set(va["customer_id"]))
    assert not (set(tr["customer_id"]) & set(te["customer_id"]))
    assert not (set(va["customer_id"]) & set(te["customer_id"]))
    assert len(tr) + len(va) + len(te) == len(dataset)


def test_model_beats_the_no_skill_baseline(trained):
    s = trained["splits"]
    p = trained["calibrator"].predict(trained["model"].predict_proba(s["X_test"])[:, 1])
    y = s["y_test"]
    pr_auc = average_precision_score(y, p)
    base = y.mean()
    assert pr_auc > base * 1.8, f"PR-AUC {pr_auc:.4f} vs base rate {base:.4f}"
    # And must NOT reach the pre-fix ~0.95, which was leakage rather than skill.
    assert pr_auc < 0.75, (
        f"PR-AUC {pr_auc:.4f} is implausibly high for this label-noise level — "
        "check whether leakage has been reintroduced"
    )


def test_is_prepaid_excluded_from_model_features():
    """It is definitionally 1 - is_cod; feeding both splits importance across a duplicate."""
    assert "is_cod" in FEATURES
    assert "is_prepaid" not in FEATURES


def test_calibration_is_strictly_monotone_so_ranking_is_preserved(trained):
    """Platt calibration must not change PR-AUC at all -- it only rescales the axis.

    This is exactly why isotonic was rejected: as a step function it is only *weakly*
    monotone, so it ties rows together and measurably degrades PR-AUC. A regression back
    to isotonic (or any binning calibrator) fails here.
    """
    s = trained["splits"]
    raw = trained["model"].predict_proba(s["X_test"])[:, 1]
    cal = trained["calibrator"].predict(raw)
    y = s["y_test"]
    assert average_precision_score(y, raw) == pytest.approx(
        average_precision_score(y, cal), abs=1e-9
    )
    # Strictly monotone => no two distinct raw scores collapse onto one calibrated score.
    assert len(np.unique(cal)) == len(np.unique(raw))
    order = np.argsort(raw)
    assert np.all(np.diff(cal[order]) >= 0)


def test_calibrated_scores_stay_below_the_fail_closed_sentinel(trained):
    """risk_score == 1.0 is reserved to mean 'the fail-closed path produced this'."""
    s = trained["splits"]
    cal = trained["calibrator"].predict(
        trained["model"].predict_proba(s["X_test"])[:, 1]
    )
    assert cal.max() < 1.0


def test_calibration_improves_brier(trained):
    from sklearn.metrics import brier_score_loss

    s = trained["splits"]
    raw = trained["model"].predict_proba(s["X_test"])[:, 1]
    cal = trained["calibrator"].predict(raw)
    y = s["y_test"]
    assert brier_score_loss(y, cal) < brier_score_loss(y, raw)


# --- section 6.4: adversarial slice --------------------------------------------------


def test_adversarial_slice_has_no_post_hoc_feature_edits():
    adv = generate_adversarial(n_rows=6_000, rng=np.random.default_rng(3))
    for col in ("order_value", "days_to_return"):
        corr = abs(np.corrcoef(adv[col].astype(float), adv["is_fraud"])[0, 1])
        assert corr < 0.15, f"{col} correlates {corr:.3f} with the adversarial label"


def test_adversarial_slice_is_genuinely_harder(trained):
    """It must be harder than in-distribution data, or it is not a stress test."""
    adv = generate_adversarial(n_rows=8_000, rng=np.random.default_rng(2026))
    X = prepare_features(adv, trained["encoder"])
    p = trained["calibrator"].predict(trained["model"].predict_proba(X)[:, 1])
    y = adv["is_fraud"].to_numpy()
    s = trained["splits"]
    p_in = trained["calibrator"].predict(
        trained["model"].predict_proba(s["X_test"])[:, 1]
    )
    assert average_precision_score(y, p) < average_precision_score(s["y_test"], p_in)


# --- section 6.3: honest validation states -------------------------------------------


def test_missing_real_data_is_never_reported_as_a_pass():
    """The exact bug from section 6.3: the absent-file branch used to return a hardcoded
    all-True dict."""
    res = load_ieee_feature_validation()
    assert res["status"] in {"not_validated", "validated", "validated_fallback"}
    if res["status"] == "not_validated":
        assert "families" not in res, "an unrun validation must not report per-feature results"
        assert "reason" in res and "how_to_enable" in res
        # Nothing in the payload may read as a success.
        assert not any(v is True for v in res.values())


def test_olist_calibration_reports_its_state_honestly():
    res = load_olist_calibration()
    assert res["status"] in {"calibrated", "not_calibrated"}
    if res["status"] == "not_calibrated":
        assert "category_share" not in res


# --- explainability ------------------------------------------------------------------


def test_explainer_returns_three_readable_reasons(trained):
    expl = RiskExplainer(trained["model"])
    X = trained["splits"]["X_test"].head(5)
    reasons = expl.top_reasons(X, k=3)
    assert len(reasons) == 5
    for row in reasons:
        assert len(row) == 3
        for r in row:
            assert r["feature"] in FEATURES
            assert r["direction"] in {"increases_risk", "decreases_risk"}
            # Must be a sentence, not a raw feature dump.
            assert len(r["reason"]) > 15
            assert "=" not in r["reason"] or r["feature"] not in r["reason"]


def test_explainer_reasons_are_ordered_by_absolute_contribution(trained):
    expl = RiskExplainer(trained["model"])
    reasons = expl.top_reasons(trained["splits"]["X_test"].head(3), k=3)
    for row in reasons:
        mags = [abs(r["contribution"]) for r in row]
        assert mags == sorted(mags, reverse=True)


def test_encoder_covers_the_fixed_taxonomy():
    enc = build_encoder()
    assert list(enc.classes_) == sorted(CATEGORIES)
