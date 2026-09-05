"""API contract, fail-closed behaviour and defense-only control tests."""
from __future__ import annotations

import copy
import json
import os

import pytest
from fastapi.testclient import TestClient

from api.app import app
from api.schemas import MAX_BATCH_SIZE
from api.service import RiskScoringService

VALID_ORDER = {
    "customer_return_rate_lt": 0.62,
    "customer_return_rate_90d": 0.75,
    "customer_account_age_days": 18,
    "customer_order_velocity_7d": 5,
    "order_value": 4899.0,
    "discount_percentage": 50.0,
    "category": "Apparel",
    "category_return_base_rate": 0.26,
    "is_prepaid": 0,
    "is_cod": 1,
    "days_to_return": 13.0,
    "same_address_returns_7d": 3,
    "same_email_returns_7d": 2,
    "payment_hash_collision": 1,
    "ip_phone_collision": 1,
    "rto_risk_score": 0.44,
}


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def _order(**overrides):
    o = copy.deepcopy(VALID_ORDER)
    o.update(overrides)
    return o


# --- basics --------------------------------------------------------------------------


def test_root(client):
    r = client.get("/")
    assert r.status_code == 200
    assert r.json()["posture"] == "defense-only, fail-closed"


def test_docs_served(client):
    assert client.get("/docs").status_code == 200
    assert client.get("/openapi.json").status_code == 200


# --- section 6.5: health check must report unhealthy honestly ------------------------


def test_health_returns_200_when_model_loaded(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "healthy"
    assert body["model_loaded"] is True
    assert body["features"] == 15


def test_health_returns_503_when_model_missing(monkeypatch):
    """Section 6.5 fix: the old handler returned 200 unconditionally, so a broken
    instance looked healthy to any orchestrator."""
    import api.app as app_module

    broken = RiskScoringService(model_dir="does/not/exist")
    assert not broken.is_ready()
    monkeypatch.setattr(app_module, "service", broken)

    with TestClient(app) as c:
        # TestClient's lifespan re-creates the service, so patch again after startup.
        monkeypatch.setattr(app_module, "service", broken)
        r = c.get("/health")
    assert r.status_code == 503
    assert r.json()["status"] == "unhealthy"
    assert r.json()["model_loaded"] is False


def test_health_does_not_leak_the_decision_threshold(client):
    """Constraint 1.1: no endpoint may reveal exact thresholds beyond what is needed to
    act on a single order. /health is a liveness probe, not an act on an order."""
    body = client.get("/health").json()
    assert "threshold" not in body
    assert body["threshold_configured"] is True


# --- scoring -------------------------------------------------------------------------


def test_score_returns_full_contract(client):
    r = client.post("/score", json=VALID_ORDER)
    assert r.status_code == 200
    b = r.json()
    assert 0.0 <= b["risk_score"] <= 1.0
    assert b["action"] in {"auto_approve", "manual_review"}
    assert len(b["reasons"]) == 3
    assert isinstance(b["threshold_used"], float)
    assert b["model_version"].startswith("rrs-")
    for reason in b["reasons"]:
        assert reason["reason"] and len(reason["reason"]) > 10


def test_action_is_consistent_with_score_and_threshold(client):
    b = client.post("/score", json=VALID_ORDER).json()
    expected = "manual_review" if b["risk_score"] >= b["threshold_used"] else "auto_approve"
    assert b["action"] == expected


def test_high_risk_order_is_flagged(client):
    b = client.post("/score", json=VALID_ORDER).json()
    assert b["action"] == "manual_review", b


def test_clean_order_scores_lower_than_a_risky_one(client):
    clean = _order(
        customer_return_rate_lt=0.03, customer_return_rate_90d=0.0,
        customer_account_age_days=1400, customer_order_velocity_7d=1,
        order_value=899.0, discount_percentage=0.0, category="Books",
        category_return_base_rate=0.06, is_prepaid=1, is_cod=0, days_to_return=2.0,
        same_address_returns_7d=0, same_email_returns_7d=0,
        payment_hash_collision=0, ip_phone_collision=0, rto_risk_score=0.05,
    )
    low = client.post("/score", json=clean).json()
    high = client.post("/score", json=VALID_ORDER).json()
    assert low["risk_score"] < high["risk_score"]


# --- section 6.7: unknown category must fail closed ----------------------------------


def test_unknown_category_fails_closed(client):
    """Section 6.7 fix: an unknown category used to be mapped to encoded index 0, so it
    silently impersonated whichever category sorted first and could be auto-approved."""
    r = client.post("/score", json=_order(category="Groceries"))
    assert r.status_code == 200
    b = r.json()
    assert b["action"] == "manual_review"
    assert b["risk_score"] == 1.0
    assert b["reasons"][0]["feature"] == "category"
    assert "Groceries" in b["reasons"][0]["reason"]
    assert b["reasons"][0]["contribution"] is None


def test_unknown_category_never_impersonates_a_known_one(client):
    """The specific old failure mode: 'Groceries' must not score like 'Apparel'."""
    unknown = client.post("/score", json=_order(category="Groceries")).json()
    apparel = client.post("/score", json=_order(category="Apparel")).json()
    assert unknown["risk_score"] != apparel["risk_score"]


def test_score_of_1_is_reserved_for_the_fail_closed_path(client):
    """A model-derived score is clamped below 1.0 so that 'certainly fraud' and 'could
    not be scored' are never conflated by a downstream consumer."""
    scored = client.post("/score", json=VALID_ORDER).json()
    assert scored["risk_score"] < 1.0
    assert scored["reasons"][0]["contribution"] is not None

    failed = client.post("/score", json=_order(category="Groceries")).json()
    assert failed["risk_score"] == 1.0
    assert failed["reasons"][0]["contribution"] is None
    assert failed["data_quality_notes"]


def test_category_is_case_sensitive_and_fails_closed(client):
    b = client.post("/score", json=_order(category="apparel")).json()
    assert b["action"] == "manual_review"
    assert b["risk_score"] == 1.0


# --- validation ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad",
    [
        {"customer_return_rate_lt": 1.5},
        {"customer_return_rate_lt": -0.1},
        {"order_value": 0},
        {"order_value": -50},
        {"discount_percentage": 120},
        {"customer_account_age_days": -1},
        {"days_to_return": -5},
        {"rto_risk_score": 1.2},
        {"payment_hash_collision": 2},
    ],
)
def test_out_of_range_values_are_rejected(client, bad):
    assert client.post("/score", json=_order(**bad)).status_code == 422


def test_contradictory_payment_flags_are_rejected(client):
    """is_prepaid is definitionally 1 - is_cod. A contradictory payload has no safe
    interpretation, so it is a 422 — which is still fail-closed: never an auto_approve."""
    assert client.post("/score", json=_order(is_prepaid=1, is_cod=1)).status_code == 422
    assert client.post("/score", json=_order(is_prepaid=0, is_cod=0)).status_code == 422


def test_missing_field_is_rejected(client):
    o = copy.deepcopy(VALID_ORDER)
    del o["order_value"]
    assert client.post("/score", json=o).status_code == 422


def test_unknown_days_to_return_sentinel_is_handled(client):
    """-1 means 'unknown'. It must be routed through the missing-value path and flagged
    as lower confidence, not treated as an extremely prompt return."""
    b = client.post("/score", json=_order(days_to_return=-1)).json()
    assert b["action"] in {"auto_approve", "manual_review"}
    assert any("days_to_return" in n for n in b["data_quality_notes"])


# --- batch ---------------------------------------------------------------------------


def test_batch_scoring(client):
    r = client.post("/score/batch", json={"orders": [VALID_ORDER, VALID_ORDER]})
    assert r.status_code == 200
    b = r.json()
    assert len(b["results"]) == 2
    assert b["summary"]["n"] == 2
    assert 0.0 <= b["summary"]["flag_rate"] <= 1.0


def test_batch_isolates_a_bad_order(client):
    """One unscoreable order must not poison the rest of the batch."""
    b = client.post(
        "/score/batch", json={"orders": [VALID_ORDER, _order(category="Groceries")]}
    ).json()
    assert b["results"][1]["action"] == "manual_review"
    assert b["results"][1]["risk_score"] == 1.0
    assert b["results"][0]["reasons"][0]["contribution"] is not None


def test_batch_size_is_capped(client):
    """Constraint 1.1: unbounded batch scoring is the cheapest way to map a decision
    boundary."""
    over = {"orders": [VALID_ORDER] * (MAX_BATCH_SIZE + 1)}
    assert client.post("/score/batch", json=over).status_code == 422


def test_empty_batch_is_rejected(client):
    assert client.post("/score/batch", json={"orders": []}).status_code == 422


# --- section 6.6: CORS ---------------------------------------------------------------


def test_cors_does_not_combine_wildcard_with_credentials(client):
    """Section 6.6 fix: allow_origins=['*'] with allow_credentials=True is rejected by
    browsers and, in some setups, reflects any origin."""
    r = client.get("/health", headers={"Origin": "https://example.com"})
    assert r.headers.get("access-control-allow-origin") == "*"
    assert "access-control-allow-credentials" not in r.headers


# --- defense-only --------------------------------------------------------------------


def test_no_endpoint_exposes_model_internals(client):
    """No feature-importance, decision-boundary or counterfactual endpoint exists."""
    paths = set(client.get("/openapi.json").json()["paths"])
    assert paths == {"/", "/health", "/score", "/score/batch"}
    for probe in ("/explain", "/importances", "/model", "/threshold", "/counterfactual"):
        assert client.get(probe).status_code == 404


def test_reasons_never_state_how_to_evade(client):
    """Reasons describe this order. They must not say how far a value would have to move
    to flip the decision."""
    b = client.post("/score", json=VALID_ORDER).json()
    blob = json.dumps(b["reasons"]).lower()
    for leak in ("threshold", "to avoid", "would need", "reduce below", "in order to pass"):
        assert leak not in blob


# --- packaging (section 6.9) ---------------------------------------------------------


def test_sample_order_json_exists_and_scores(client):
    """Section 6.9: sample_order.json was referenced in the documented curl command but
    nothing ever created it."""
    assert os.path.exists("sample_order.json")
    with open("sample_order.json", encoding="utf-8") as f:
        payload = json.load(f)
    assert client.post("/score", json=payload).status_code == 200
