# PROJECT_SPEC.md — Return Risk Scorer
### Razorpay AI Buildathon 2026 — Track 02: AI Risk Manager

This is the complete, authoritative specification. It supersedes an earlier draft of this
project that had real issues — this document already incorporates the diagnosed root
causes and validated fixes for each, plus a set of enhancements. Build against this
document, not against assumptions about what a "typical" version of this project looks
like. Where anything here is ambiguous, make a reasonable, defensible engineering
decision, state the assumption in the code/docs, and keep moving — don't stall on
clarifying questions unless something is genuinely consequential and unresolvable from
context.

## 0. Context you should know

- This is a submission for Razorpay's AI Buildathon — a hiring funnel for a paid **AI
  Builder Intern** role (₹75,000/month, 6–12 months, in-person Bangalore). Selection has
  **no resume screen, no aptitude test, no GD**: a public GitHub repo, a 5-minute pitch
  video, and an architecture walkthrough in a panel interview.
- **Track 02's evaluation bar, verbatim in spirit:** a working detector/verifier for one
  loss type, judged on *honest* precision/recall including false-positive cost, and it
  must be **strictly defense-only** — anything offense-capable is disqualified.
- Razorpay already runs an internal return-risk product ("RTO Shield") and a fraud/risk
  foundation model ("Vulcan") in production. Don't claim to be inventing something novel
  — the differentiator here is **rigor and honesty of evaluation**, not novelty.
- Because reviewers read the code and ask about specific choices, every claimed metric
  must be real and reproducible, and every design choice should have a stated reason.

## 0.1 What's fixed vs. what's a starting suggestion

Treat the following as genuine, non-negotiable requirements: Track 02's evaluation bar
(§0), the constraints in §1, the underlying *principle* behind §6.1 (no feature used to
construct the label may also be fed to the model, and nothing may be rewritten after the
label is realized), and the general output shape (an explainable score + bounded action
+ reasons).

Everything else — the specific tech stack in §3, the exact file layout in §4, the choice
of XGBoost/FastAPI/Streamlit or any other named tool, and even the specific return-fraud
framing itself — is a **starting point carried over from an earlier draft, not a
mandate**. This spec was written to fix and extend that earlier draft, which is why it
inherited its shape. If a different technology, structure, or a different sub-problem
within Track 02 would genuinely better satisfy the actual requirements above, use that
judgment instead of matching the draft. The only obligation is documenting the reasoning
in ARCHITECTURE.md (§7.3) — a good, justified deviation is worth more in a panel
interview than dutifully preserving an earlier draft's specific choices. Do not treat any
example code in §6 as something to copy verbatim if you have a better way to satisfy the
principle it's demonstrating.

## 1. Non-negotiable constraints

1. **Defense-only.** The system detects and explains risk. It must never expose any
   capability that helps an attacker calibrate evasion (e.g., no endpoint that reveals
   exact decision thresholds beyond what's needed to act on a single legitimate order, no
   capability to generate/optimize fraud patterns).
2. **Fail-closed by default.** Any error, unrecognized input, or low-confidence state
   routes to `manual_review`, never to a silent `auto_approve`.
3. **Honest metrics, always.** Never adjust the data generator, thresholds, or reporting
   to make a number look better. If a fix changes a metric for the worse, report the new
   number and explain why it's more credible than the old one.
4. **Everything must actually run.** Don't hand back code you haven't executed. Run the
   full pipeline (data → train → evaluate → API → tests) yourself and confirm it
   completes cleanly before presenting anything as finished.

## 2. Project overview

**What it does:** given an order and a return request, output a fraud-risk score
(0–1), a bounded action (`auto_approve` or `manual_review`), and the top 3 human-readable
risk drivers (via SHAP).

**Why it matters:** return fraud and RTO (return-to-origin) losses are a meaningful
margin problem for Indian D2C e-commerce, COD orders carry materially higher RTO loss
than prepaid, and manual review has a real per-order cost — so the system needs to be
useful (catch real fraud), cheap to run against (don't flood merchants with false
positives), and explainable (a merchant ops team needs to trust and act on it).

**Differentiators to build in:** India-specific COD/RTO patterns, a genuinely
adversarial held-out test slice, a cost-sensitivity analysis (not just one threshold),
SHAP explainability with merchant-readable reasons, and a defense-only, fail-closed
design — all backed by an automated check that the model isn't cheating via leakage (see
§6.1 and §7.1).

## 3. Tech stack

| Layer | Technology | Purpose |
|---|---|---|
| Language | Python 3.10+ | |
| Data | pandas, numpy | manipulation |
| ML | XGBoost, scikit-learn | primary model, preprocessing, metrics |
| Explainability | SHAP | feature attribution |
| Visualization | matplotlib, seaborn | cost curves, calibration plots |
| API | FastAPI, Uvicorn, Pydantic | REST service |
| Dashboard | Streamlit | reviewer-facing demo (§7.6) |
| Testing | Pytest, httpx | unit/integration/leakage tests |
| CI | GitHub Actions | run tests on every push (§7.8) |
| Serialization | Joblib | model persistence |
| Deployment | Docker, Hugging Face Spaces | see §8 |

## 4. Project structure

```
return-risk-scorer/
├── README.md                  # top-level, reviewer-facing (§7.9)
├── PROJECT_GUIDE.md            # required companion doc — see §10
├── ARCHITECTURE.md             # decision log — see §7.3
├── SAFETY.md                   # defense-only rationale — see §7.5
├── requirements.txt
├── requirements-lock.txt       # pip freeze of the exact tested versions
├── Dockerfile                  # Hugging Face deployment — see §8
├── run.py
├── .github/workflows/ci.yml    # see §7.8
├── data/
│   ├── synthetic.py            # §6.1 — corrected generator
│   ├── adversarial.py          # §6.4 — corrected adversarial slice
│   └── real_calibration.py
├── models/
│   ├── train.py                # §6.2 — corrected XGBoost usage
│   └── explainer.py
├── evaluate/
│   ├── metrics.py
│   ├── cost_curve.py
│   ├── sensitivity.py
│   ├── calibration.py          # §7.2 — new
│   └── leakage_test.py         # §7.1 — new, also runs as a pytest
├── api/
│   ├── app.py                  # §6.5, §6.6 — corrected health check, CORS
│   ├── schemas.py
│   └── service.py              # §6.7 — corrected category fallback
├── dashboard/
│   └── app.py                  # §7.6 — new
├── tests/
│   ├── test_api.py
│   ├── test_model.py
│   └── test_leakage.py         # §7.1
├── scripts/
│   ├── download_olist.py
│   └── download_ieee.py
├── sample_order.json           # §6.9 — was referenced but never created
└── outputs/                    # generated: cost_curve.png, cost_sensitivity.png/csv,
                                 # calibration.png, model card metrics
```

## 5. API contract (unchanged from the original spec — this part was correct)

`OrderRequest` (Pydantic): `customer_return_rate_lt` (float, 0–1), `customer_return_rate_90d`
(float, 0–1), `customer_account_age_days` (int, ≥0), `customer_order_velocity_7d` (int,
≥0), `order_value` (float, >0, INR), `discount_percentage` (float, 0–100), `category`
(str, one of Apparel/Electronics/Home/Beauty/Books), `category_return_base_rate` (float,
0–1), `is_prepaid` (int, 0/1), `is_cod` (int, 0/1), `days_to_return` (float, ≥-1),
`same_address_returns_7d` (int, ≥0), `same_email_returns_7d` (int, ≥0),
`payment_hash_collision` (int, 0/1), `ip_phone_collision` (int, 0/1), `rto_risk_score`
(float, 0–1).

Endpoints: `GET /`, `GET /health`, `POST /score`, `POST /score/batch`, `GET /docs`.

## 6. Must-fix issues — root cause and validated fix for each

These were found by reconstructing the previous version of this project and actually
running it. Treat these as the highest priority — correctness before any enhancement in
§7.

### 6.1 Data generator leaks the label into the features (critical)

**Root cause:** the previous generator built `fraud_prob` directly from `order_value`,
`is_cod`, `discount_percentage`, and `days_to_return` — then, after drawing `is_fraud`
from that probability, ran a block that made those *same* features more extreme
specifically for the fraud rows. A derived `rto_risk_score` feature then re-combined the
same signals again. Verified impact: with all features, PR-AUC reached ~0.95; dropping
just the features used to build the label collapsed it to ~0.32, barely above the
no-skill baseline (~0.22 at that fraud rate). The model was reverse-engineering the label
formula, not learning fraud patterns.

**Required design:** fraud risk must come from a **latent, unobserved customer trait**
that *tilts* the distributions observable features are drawn from — never from a
deterministic or post-hoc-amplified function of the features the model will see.
Concretely:

```python
# is_fraudster is a latent trait, decided before any order-level features.
if is_fraudster:
    is_cod = np.random.choice([0, 1], p=[0.45, 0.55])          # tilted, not deterministic
    order_value = max(100, min(50000, np.random.gamma(2, 900) + 300))
    discount = np.random.choice([0,5,10,20,30,50,70],
                                 p=[0.10,0.15,0.15,0.20,0.15,0.15,0.10])
else:
    is_cod = np.random.choice([0, 1], p=[0.6, 0.4])
    order_value = max(100, min(50000, np.random.gamma(2, 750) + 300))
    discount = np.random.choice([0,5,10,20,30,50,70],
                                 p=[0.20,0.25,0.20,0.15,0.10,0.07,0.03])
is_prepaid = 1 - is_cod

# The label follows from the latent trait (+ mild independent noise), not from
# thresholds on features that are also model inputs.
fraud_prob = min(0.55 * is_fraudster + 0.08 * (account_age[cust_id] < 30) + 0.04, 0.95)
is_fraud = 1 if (is_returned and np.random.random() < fraud_prob) else 0
# Do NOT add any block that rewrites order_value/discount/is_cod/days_to_return
# after is_fraud is known. That is the leak.
```

Validated result of this design: PR-AUC ≈ 0.35 against a ~12% base rate (a real ~3x
lift), with feature importance spread across return-rate history, account age, and
network features rather than dominated by order_value/is_cod/discount. **Expect and
report a number in this range — do not chase 0.86.** A real 0.3–0.5 PR-AUC, honestly
derived, is the correct target, not a shortfall.

Either drop `rto_risk_score` as a feature entirely (it's a redundant linear
recombination of other raw features using near-identical thresholds to the old label
formula), or, if you keep it, derive it from a genuinely independent signal so it isn't
just restating the label logic.

**Recommended data sourcing — three distinct roles, don't conflate them:**

1. *Realistic order/category distributions.* Calibrate against the Olist Brazilian
   E-commerce dataset (Kaggle: `olistbr/brazilian-ecommerce`, ~100K real orders). Source
   it via the Kaggle API/download, not an unverified raw-GitHub-URL mirror.
2. *Validating that the engineered feature types genuinely carry fraud signal in
   reality.* Validate against the IEEE-CIS Fraud Detection dataset (Kaggle competition
   `ieee-fraud-detection`, ~590K real labeled transactions — needs a free Kaggle account
   + API token; document that setup step explicitly, don't silently no-op when the file
   is absent, per §6.3). If that access proves friction-y, the smaller ULB Credit Card
   Fraud dataset (Kaggle: `mlg-ulb/creditcardfraud`, 284,807 real transactions, 492
   frauds) is a simpler fallback for the same purpose.
3. *The actual India COD/RTO return-fraud labels.* There is no real, public,
   India-specific dataset for this — every public "return prediction" dataset available
   is either explicitly synthetic, generic (not fraud-labeled), or not India-specific.
   Use the corrected generator above and say so plainly in the model card. This is a
   genuine data gap being honestly disclosed, not a corner being cut.

### 6.2 XGBoost training call will crash on the required library version

**Root cause:** `early_stopping_rounds` was passed to `.fit()`. On xgboost ≥2.0 (which
`requirements.txt` requires) this raises `TypeError: XGBClassifier.fit() got an
unexpected keyword argument 'early_stopping_rounds'` — verified directly against
xgboost 3.4.1.

**Fix:** pass it to the constructor, not `.fit()`:

```python
model = xgb.XGBClassifier(
    n_estimators=200, max_depth=6, learning_rate=0.05,
    scale_pos_weight=scale_pos, random_state=42,
    eval_metric='aucpr',
    early_stopping_rounds=20,          # constructor, not .fit()
)
model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
```

### 6.3 A "real-data validation" check silently reports success when it never ran

**Root cause:** `load_ieee_feature_validation()` is meant to confirm the model's feature
types appear in the real IEEE-CIS fraud dataset. That dataset requires a Kaggle account
and API token to download, and no such setup is documented anywhere — so the file will
almost always be absent, and the function's `except` branch returns a hardcoded
all-`True` dict regardless. This makes an unvalidated claim look validated, which is
exactly the wrong kind of mistake for a track whose bar is "honest metrics."

**Fix:** make the absent-file case a distinct, honestly labeled state:

```python
def load_ieee_feature_validation() -> Dict[str, Any]:
    ieee_path = 'data/raw/ieee_fraud.csv'
    if not os.path.exists(ieee_path):
        logger.warning("IEEE-CIS file not found — validation SKIPPED (not run)")
        return {'status': 'not_validated', 'reason': 'ieee_fraud.csv not present'}
    # only return real per-feature True/False checks once the file is confirmed present
```

Either implement and document the Kaggle download step so this can genuinely run, or
keep it explicitly marked "not validated" in the model card rather than omitted.

### 6.4 Adversarial test slice uses a subtler version of the same leak

**Root cause:** `is_fraud` in the adversarial generator is drawn first (flat 15% rate),
then `order_value`/`days_to_return` are shifted upward afterward for fraud rows — the
same causal-ordering mistake as §6.1, just weaker. This means the "adversarial gap"
metric partly just measures how much weaker that particular leak was made, not real
generalization to novel fraud patterns.

**Fix:** rebuild it with the same latent-trait, tilted-distribution approach as §6.1 —
draw a `looks_genuine_fraudster` trait, tilt distributions subtly (this slice is
*supposed* to look close to genuine), and derive the label from the trait, never from
post-hoc feature edits.

### 6.5 Health check always returns HTTP 200, even when unhealthy

**Fix:**
```python
from fastapi.responses import JSONResponse

@app.get("/health")
async def health():
    if service is None or not service.is_ready():
        return JSONResponse(status_code=503, content={"status": "unhealthy", "model_loaded": False})
    return {"status": "healthy", "model_loaded": True, "threshold": float(service.threshold),
            "features": len(service.features), "categories": service.category_names}
```

### 6.6 CORS: wildcard origin combined with credentials

**Root cause:** `allow_origins=["*"]` with `allow_credentials=True` is a known-bad
combination (browsers reject it, or worse, some setups reflect any origin). There's no
cookie/session auth in this API, so:

```python
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=False,
                    allow_methods=["*"], allow_headers=["*"])
```

### 6.7 Unrecognized category silently falls back instead of failing closed

**Root cause:** an unknown `category` value was mapped to whatever encoded index `0`
happens to be — silently impersonating a specific known category instead of triggering
the fail-closed path used elsewhere for scoring errors. **Fix:**

```python
if category not in self.encoder.classes_:
    return {'risk_score': 1.0, 'action': 'manual_review',
            'reasons': [{'reason': f'Unrecognized category "{category}" — routed to manual review',
                         'contribution': None, 'feature': 'category'}],
            'threshold_used': self.threshold}
```

### 6.8 No version ceiling, already caused one break

**Fix:** don't guess at upper bounds — generate `requirements-lock.txt` via `pip freeze`
from what you actually tested (xgboost 3.4.1 / shap 0.52.0 / scikit-learn 1.8.0 / pandas
3.0.2 / numpy 2.4.4 are all confirmed working once §6.2 is applied), and build the Docker
image from the lock file so a reviewer's build resolves to what you verified, not
whatever is newest on the day they build it.

### 6.9 Minor packaging gaps

- `sample_order.json` is referenced in the documented `curl` command but nothing writes
  it — add it as a static file at the repo root.
- Any architecture diagrams should be plain Mermaid inside `ARCHITECTURE.md` /
  `README.md` (renders natively on GitHub) — don't rely on a PDF export, which won't
  render Mermaid at all. The deliverable is a GitHub repo, not a PDF.

## 7. Enhancements — required, not optional, to clear "exceptional"

Only attempt these after everything in §6 is correct and passing. If something has to
give under time pressure, correctness (§6) wins over polish (§7) — but the goal is both.

### 7.1 Automated leakage smell-test (make §6.1's fix permanent and visible)

Add `evaluate/leakage_test.py`, runnable both standalone and as `tests/test_leakage.py`:
- Train the model, record feature importances. **Assert no single feature's importance
  exceeds ~25%** of total.
- Retrain after dropping the top-3 features by importance. **Assert PR-AUC does not drop
  by more than ~50% relative** to the full-feature model (i.e., the model isn't
  dependent on a tiny feature subset).
- Run this in CI (§7.8) on every push. This turns the earlier bug into a demonstrable,
  ongoing rigor claim: "we test for and rule out label leakage automatically."

### 7.2 Probability calibration reporting

The spec asks for a *risk score*, not just a binary action — so its calibration matters.
Add `evaluate/calibration.py`: compute the Brier score and a reliability
diagram (predicted-probability bucket vs. actual fraud rate in that bucket). Include
both in the model card and in `outputs/`.

### 7.3 ARCHITECTURE.md — a real decision log

Document the 6–8 decisions a panel is likely to ask about, each as (decision → why →
what was rejected): e.g., latent-trait data generation over feature-conditioned labels
(§6.1), `scale_pos_weight` over SMOTE (SMOTE would interpolate synthetic points on top of
already-synthetic data), XGBoost over a neural net (tabular data, small feature count,
need for SHAP-based explainability), fail-closed defaults, the specific cost assumptions
(§7.4).

### 7.4 Ground the cost assumptions, don't just assert them

State an explicit, labeled-as-illustrative reasoning chain for FP (~₹50) and FN (~₹500)
costs — e.g., FN ≈ unrecovered cost-of-goods + return shipping + support handling for a
fraudulent return; FP ≈ support/friction cost of wrongly flagging a genuine customer.
Say plainly that a real merchant would calibrate these from their own numbers — the
point is showing the reasoning, not claiming precision you don't have.

### 7.5 SAFETY.md — explicit defense-only rationale

State plainly what the system does and does not expose: no endpoint reveals exact
decision thresholds beyond what's needed to act on one legitimate order; no capability to
batch-probe the model to reverse-engineer its decision boundary; no functionality that
generates or optimizes evasive transaction patterns. This directly addresses Track 02's
explicit disqualification condition — make it easy for a reviewer to confirm.

### 7.6 A reviewer-facing dashboard (this is what the pitch video should show)

`dashboard/app.py` (Streamlit): upload a CSV batch or fill in one order manually, see
the risk score, a SHAP waterfall for the top reasons, and where that order sits on the
cost curve. A live, visual demo is far more compelling in a 5-minute pitch video than a
terminal `curl` command.

### 7.7 Business-impact translation

Add a small, clearly-labeled "illustrative impact" calculation: at a chosen operating
point, translate precision/recall/cost into something like "for every 10,000 returns
processed, X are flagged, ~₹Y saved vs. the approve-all baseline" — grounded in the
project's own stated 8–12% margin-erosion context, explicitly marked as order-of-
magnitude, not a guarantee.

### 7.8 CI

`.github/workflows/ci.yml`: run `pytest tests/ -v` (including `test_leakage.py`) on every
push. Cheap to add, signals real engineering discipline.

### 7.9 A polished top-level README (distinct from the internal model card)

2–3 sentence pitch, badges (build status), quickstart, a link to the live Hugging Face
Space once deployed, and a short results summary (real PR-AUC, cost reduction %, from
the corrected pipeline). This is often the first and only thing a reviewer reads before
deciding whether to dig further — it should not read like internal documentation.

## 8. Hugging Face Spaces deployment

- **Docker SDK**, not Gradio/Streamlit templates — add a `Dockerfile` and a `README.md`
  with HF's YAML frontmatter (`sdk: docker`, `app_port: 7860`).
- **Listen on port 7860**, not 8000 — Spaces expects this by default.
- **Deploy the trained artifacts, not the training pipeline.** Free-tier Spaces give a
  writable `/tmp` only at runtime (no persistent disk). Either commit the small `.pkl`
  files directly, or run `run.py` as a Docker **build-time** `RUN` step (the build
  filesystem is writable; the running container's isn't). The Space only needs
  `models/*.pkl` + `api/` + `dashboard/` — not the full `data/` generation pipeline.
- **No GPU needed** — free CPU Basic (2 vCPU / 16GB RAM) is comfortably enough for
  XGBoost + SHAP inference on 16 tabular features.
- Run as non-root (`RUN useradd -m -u 1000 user`, then `USER user`), matching HF's
  recommended pattern.
- Public Spaces mean public code and logs — don't commit secrets (this project needs
  none). This is actually good for judging: reviewers can inspect and hit the live API
  directly.

## 9. Deliverables checklist

- [ ] Complete, runnable codebase per §4, with every fix in §6 applied
- [ ] Every enhancement in §7 implemented
- [ ] `python run.py` completes end-to-end with no errors, using `requirements-lock.txt`
- [ ] `pytest tests/ -v` passes, including the new leakage test
- [ ] `uvicorn api.app:app --reload` serves `/docs`, `/health` (correct status codes),
      `/score`, `/score/batch`
- [ ] `streamlit run dashboard/app.py` runs the reviewer-facing demo
- [ ] `Dockerfile` builds and runs correctly on port 7860
- [ ] Model card in `README.md`/`models/` reports **real, post-fix** metrics — PR-AUC,
      precision, recall, cost/order, Brier score, adversarial recall gap — with no
      number that isn't reproducible by re-running the pipeline
- [ ] `ARCHITECTURE.md`, `SAFETY.md`, `sample_order.json` present
- [ ] `PROJECT_GUIDE.md` present — see §10

## 10. Required companion document: PROJECT_GUIDE.md

Produce this alongside the code, written for the person who will actually submit this
(not another engineer). It must include, in this order:

1. **What was built** — plain-language summary of the system and what changed from the
   original draft (the leakage fix, the crash fix, the fake-validation fix, and the
   enhancements), including the real before/after metrics.
2. **Architecture at a glance** — a short diagram or bullet flow of data → model →
   API → dashboard → deployment.
3. **Local setup and run** — exact, copy-pasteable commands, in order, from a clean
   clone to a running API and dashboard, using `requirements-lock.txt`.
4. **Testing** — exact commands to run the test suite including the leakage test, and
   what a passing result looks like.
5. **Deploying to Hugging Face Spaces** — exact step-by-step: create the Space (Docker
   SDK), what to push, how to verify it's live, how to hit the live API.
6. **Submission checklist**, mapped to the buildathon's actual process: making the
   GitHub repo public, what the README should show a first-time reviewer, what to cover
   in the 5-minute pitch video (grounded in what's actually in this repo — the real
   metrics, the leakage-test as a rigor signal, the live HF demo), and what to be ready
   to explain in the architecture/panel round (anticipate: "why does dropping the top
   features tank performance so much less dramatically now than a naive version would?"
   and "how did you pick your cost assumptions?").
7. **Known limitations**, stated honestly (synthetic data, illustrative cost
   assumptions, IEEE-CIS validation status) — matching what's actually in the model
   card, not softened.

## 11. Priority order if time is short

1. §6 in full (correctness) — nothing else matters if this isn't right
2. §7.1 and §7.5 (leakage test, safety doc) — directly defend against the two things
   most likely to sink a Track 02 submission specifically
3. §8 (Hugging Face deployment) — a live demo is high-leverage for the pitch video
4. §7.6 (dashboard) — second-highest leverage for the pitch video
5. Remaining §7 items and §10 (PROJECT_GUIDE.md) — do this regardless of time pressure,
   it's the deliverable that turns working code into a submittable package
