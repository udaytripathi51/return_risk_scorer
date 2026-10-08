"""The leakage test suite as pytest, in both directions.

Runs in CI on every push. A change that introduces label leakage -- into the generator,
the split, or the feature assembly -- fails the build instead of quietly inflating
PR-AUC. And because a test that cannot fail proves nothing, the same suite must flag a
deliberately leaky copy of the data (the positive control).

The datasets are smaller here than in the standalone run (`python -m evaluate.leakage_test`)
so CI stays fast. Each still trains three full models: the real one, the top-3-ablated
one, and the shuffled-label control.
"""
from __future__ import annotations

import numpy as np
import pytest

from data.synthetic import generate_dataset
from evaluate.leakage_test import (
    MAX_RELATIVE_PR_AUC_DROP,
    MAX_SHUFFLE_LIFT,
    MAX_SINGLE_FEATURE_IMPORTANCE,
    make_leaky_control,
    run_leakage_checks,
)

N_CUSTOMERS_CI = 25_000


@pytest.fixture(scope="module")
def checks():
    return run_leakage_checks(n_customers=N_CUSTOMERS_CI, seed=42, verbose=True)


@pytest.fixture(scope="module")
def leaky_checks():
    """The same suite run on a deliberately leaky copy of the data."""
    data = generate_dataset(n_customers=N_CUSTOMERS_CI, rng=np.random.default_rng(42))
    leaky = make_leaky_control(data, rng=np.random.default_rng(44))
    return run_leakage_checks(n_customers=N_CUSTOMERS_CI, df=leaky, verbose=False)


def test_no_single_feature_dominates(checks):
    """A leaked label shows up as one feature swallowing the model."""
    c = checks["concentration"]
    assert c["max_single_feature_importance"] <= MAX_SINGLE_FEATURE_IMPORTANCE, (
        f"{c['top_feature']} holds {c['max_single_feature_importance']:.1%} of gain "
        f"importance (cap {MAX_SINGLE_FEATURE_IMPORTANCE:.0%}). Suspect label leakage: "
        f"a feature this dominant usually means the label is a function of it."
    )


def test_model_survives_dropping_top_3_features(checks):
    """A model inverting a leaked label formula collapses towards the base rate when
    its top features are withheld (the positive control below shows it). A model
    aggregating genuinely distributed evidence degrades gracefully instead."""
    a = checks["top3_ablation"]
    assert a["relative_drop"] <= MAX_RELATIVE_PR_AUC_DROP, (
        f"Dropping {a['dropped_features']} cut PR-AUC from {a['full_pr_auc']} to "
        f"{a['reduced_pr_auc']} ({a['relative_drop']:.1%} relative). The model depends "
        f"on a tiny feature subset — the signature of leakage."
    )


def test_shuffled_labels_collapse_to_baseline(checks):
    """Harness-level negative control: with permuted labels there is nothing to learn, so
    PR-AUC must fall to the base rate. If it does not, the train/evaluate harness itself
    is leaking (for example, scoring rows the model trained on). It cannot catch a
    feature computed from the label; the two checks above target that."""
    s = checks["label_shuffle_control"]
    assert s["lift_over_base_rate"] <= MAX_SHUFFLE_LIFT, (
        f"A shuffled-label model scored PR-AUC {s['shuffled_pr_auc']} against a base "
        f"rate of {s['base_rate']} ({s['lift_over_base_rate']:.2f}x). With random labels "
        f"there is nothing to learn, so this indicates leakage in the pipeline itself."
    )


def test_all_leakage_checks_pass(checks):
    assert checks["all_passed"], checks


def test_suite_flags_a_deliberately_leaky_dataset(leaky_checks):
    """Positive control. A test that cannot fail proves nothing when it passes: on data
    whose label is computed from observable features, the concentration check and the
    top-3 ablation must both fail."""
    assert leaky_checks["top3_ablation"]["full_pr_auc"] > 0.85, leaky_checks["top3_ablation"]
    assert not leaky_checks["concentration"]["passed"], leaky_checks["concentration"]
    assert not leaky_checks["top3_ablation"]["passed"], leaky_checks["top3_ablation"]
    assert not leaky_checks["all_passed"]
