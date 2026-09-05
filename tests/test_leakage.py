"""Leakage smell test as a pytest (PROJECT_SPEC.md sections 7.1 and 7.8).

Runs in CI on every push. A change that reintroduces label leakage -- into the generator,
the split, or the feature assembly -- fails the build instead of quietly inflating
PR-AUC.

The dataset is smaller here than in the standalone run (`python -m evaluate.leakage_test`)
so CI stays under a couple of minutes. It still trains three full models: the real one,
the top-3-ablated one, and the shuffled-label control.
"""
from __future__ import annotations

import pytest

from evaluate.leakage_test import (
    MAX_RELATIVE_PR_AUC_DROP,
    MAX_SHUFFLE_LIFT,
    MAX_SINGLE_FEATURE_IMPORTANCE,
    run_leakage_checks,
)

N_CUSTOMERS_CI = 25_000


@pytest.fixture(scope="module")
def checks():
    return run_leakage_checks(n_customers=N_CUSTOMERS_CI, seed=42, verbose=True)


def test_no_single_feature_dominates(checks):
    """A leaked label shows up as one feature swallowing the model."""
    c = checks["concentration"]
    assert c["max_single_feature_importance"] <= MAX_SINGLE_FEATURE_IMPORTANCE, (
        f"{c['top_feature']} holds {c['max_single_feature_importance']:.1%} of gain "
        f"importance (cap {MAX_SINGLE_FEATURE_IMPORTANCE:.0%}). Suspect label leakage: "
        f"a feature this dominant usually means the label is a function of it."
    )


def test_model_survives_dropping_top_3_features(checks):
    """The pre-fix model went ~0.95 -> ~0.32 (66% drop) when its top features were
    withheld, because it was inverting the label formula. A model aggregating genuinely
    distributed evidence degrades gracefully instead."""
    a = checks["top3_ablation"]
    assert a["relative_drop"] <= MAX_RELATIVE_PR_AUC_DROP, (
        f"Dropping {a['dropped_features']} cut PR-AUC from {a['full_pr_auc']} to "
        f"{a['reduced_pr_auc']} ({a['relative_drop']:.1%} relative). The model depends "
        f"on a tiny feature subset — the signature of leakage."
    )


def test_shuffled_labels_collapse_to_baseline(checks):
    """Pipeline-level control: with permuted labels there is nothing to learn, so PR-AUC
    must fall to the base rate. If it does not, the split/encoding/calibration plumbing
    is leaking, independently of the generator."""
    s = checks["label_shuffle_control"]
    assert s["lift_over_base_rate"] <= MAX_SHUFFLE_LIFT, (
        f"A shuffled-label model scored PR-AUC {s['shuffled_pr_auc']} against a base "
        f"rate of {s['base_rate']} ({s['lift_over_base_rate']:.2f}x). With random labels "
        f"there is nothing to learn, so this indicates leakage in the pipeline itself."
    )


def test_all_leakage_checks_pass(checks):
    assert checks["all_passed"], checks
