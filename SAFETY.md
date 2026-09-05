# SAFETY.md — defense-only rationale

Track 02 disqualifies anything offense-capable. This document states plainly what the
system does and does not expose, so a reviewer can confirm it quickly rather than infer it
from the code. Every claim below names the file that enforces it.

## The one-line claim

**This system detects and explains return-fraud risk. It has no capability to generate,
optimise, or test evasive patterns, and it exposes no interface that meaningfully helps an
attacker locate its decision boundary.**

## What is exposed, and why each is necessary

| Exposed | Where | Why it is necessary |
|---|---|---|
| `risk_score` (0–1) for one submitted order | `POST /score` | The merchant must triage this order. §2 requires a score, not just a binary. |
| `action` (`auto_approve` / `manual_review`) | `POST /score` | The bounded decision the system exists to produce. |
| Top-3 reasons for **this** order | `POST /score` | A merchant ops team cannot act on a score it cannot explain. |
| `threshold_used` for **this** order | `POST /score` | Needed to interpret this one decision — see the carve-out below. |
| Liveness/readiness, feature **count**, category list | `GET /health` | Operational necessity for deployment. |

## What is deliberately NOT exposed

| Not exposed | Enforced in | Rationale |
|---|---|---|
| Global feature importances | no such endpoint; `tests/test_api.py::test_no_endpoint_exposes_model_internals` | Ranking which signals matter most is a direct map of what to suppress first. |
| The decision threshold as a system-wide value | `api/app.py` — `/health` returns `threshold_configured: true` | An unauthenticated liveness probe has no need for it. |
| Any counterfactual / "what would make this approve" | no such endpoint | The single most directly evasion-enabling feature a risk API can have. |
| Unbounded batch scoring | `api/schemas.py` — `MAX_BATCH_SIZE = 100` | Bulk probing is the cheapest way to reconstruct a decision surface. |
| Unlimited request volume | `api/app.py` — 120 requests / 60s per client | Makes iterative boundary-mapping slow and visible in logs. |
| Model internals, weights, or trees | not served | Full white-box access trivially yields evasion strategies. |
| Any pattern-generation or optimisation capability | not implemented anywhere | This is the disqualifying capability. It does not exist in this codebase. |

## The threshold carve-out, stated explicitly

Constraint §1.1 forbids revealing exact decision thresholds *"beyond what's needed to act
on a single legitimate order."* That phrasing contains a deliberate carve-out, and this
system sits on both sides of it consciously:

- **`/score` returns `threshold_used`.** This is inside the carve-out. The merchant needs
  it to interpret the one order they asked about. Critically, it leaks **nothing**: the
  same response already contains `risk_score` and `action`, and any attacker who can read
  both can already infer the threshold's position from a single call. Withholding it would
  cost the merchant real interpretability and buy no security.

- **`/health` does not return the threshold.** This is outside the carve-out. A liveness
  probe is not an act on an order, so publishing a global decision constant there is
  disclosure with no operational justification.

This is a **deliberate deviation from the example snippet in `PROJECT_SPEC.md` §6.5**,
which places `threshold` in the health body. §0.1 makes the constraint binding and the
example code advisory; the constraint wins. See `ARCHITECTURE.md` decision 11.

## Explanations are descriptive, never prescriptive

SHAP reasons (`models/explainer.py`) describe what *this* order looks like:

> "Returns 62% of lifetime orders, above the ~12% norm"

They never state the decision boundary, never quantify how far a value would have to move
to flip the decision, and never rank which signal is cheapest to defeat.
`tests/test_api.py::test_reasons_never_state_how_to_evade` asserts that reason text
contains no threshold or "would need to" phrasing.

The `contribution` field carries SHAP values in raw log-odds margin units. This is
attribution for a decision already made — it does not tell an attacker what to change,
because the tilts are distributional, not rule-based: there is no threshold to step over.

## Fail-closed is a safety property, not just error handling

There is exactly one route to `auto_approve`: a fully-parsed order whose model score came
back below the threshold. Every other path returns `risk_score = 1.0` and `manual_review`:

- unrecognised `category` → review (`api/service.py`)
- model artefacts missing or unloadable → review, and `/health` reports **503**
- any exception during feature assembly, inference or explanation → review
- contradictory `is_prepaid`/`is_cod` → HTTP 422 (never an approval)
- batch over 100, or empty → HTTP 422

An attacker who finds a way to *crash* the scorer therefore gains nothing: the result of
breaking this system is that everything goes to a human, which is the safe direction.
`risk_score = 1.0` is reserved exclusively for this path, so a downstream consumer can
always distinguish "certainly fraud" from "could not be assessed".

## Data handling

- The generator produces **entirely synthetic** data. No real customer data is present
  anywhere in this repository.
- Identity signals are consumed as **pre-hashed collision booleans**
  (`payment_hash_collision`, `ip_phone_collision`) — never raw emails, phone numbers,
  payment instruments or IP addresses. The service could not re-identify a person from its
  inputs even if asked to.
- No secrets are required by this project, which matters because Hugging Face Spaces are
  public: code, logs and all.

## What this system cannot do, and does not claim to

- It cannot adjudicate. It routes to a human; the human decides.
- It does not resist an adapted abuser well, and says so with a number: recall falls from
  **0.754 to 0.331** (a **56.2%** relative drop) on the adversarial slice. That gap is
  published in the model card rather than minimised, because a merchant deciding how much
  to trust the auto-approve path needs it.
- It has not been audited for demographic fairness. The generator carries no demographic
  attributes, so such an audit would be vacuous here — but a real deployment must run one
  before going live.
