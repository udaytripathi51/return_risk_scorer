# PROJECT_GUIDE.md — runbook

How to set up, test, deploy and verify Return Risk Scorer, and how to reproduce its numbers.
For what the system does and why, start with the [README](README.md) and
[ARCHITECTURE.md](ARCHITECTURE.md). Every number here comes from `outputs/metrics.json`
unless a section says otherwise.

## 1. What runs where

```text
data/synthetic.py      latent traits -> tilted features -> label (33,331 return requests)
data/adversarial.py    adapted-abuser stress slice, same causal design
models/train.py        grouped 60/20/20 split -> XGBoost (early stopping on val)
                       -> Platt calibration (val) -> cost-minimising threshold (val)
evaluate/*             test metrics, calibration, cost curve + sensitivity, adversarial
                       slice, leakage suite + positive control, diagnostics
api/                   FastAPI: /, /health, /score, /score/batch, /docs
                       RiskScoringService: fail-closed scoring + SHAP reasons
dashboard/app.py       Streamlit reviewer UI on the same RiskScoringService
.github/workflows/     CI: the full test suite and the leakage suite on every push
```

## 2. Local setup

Python 3.13 is what the lock file was produced with.

```bash
git clone https://github.com/udaytripathi51/return_risk_scorer.git
cd return_risk_scorer
python -m venv .venv
# Windows:        .venv\Scripts\activate
# macOS / Linux:  source .venv/bin/activate
pip install -r requirements-lock.txt

python run.py                      # data -> train -> evaluate -> outputs/ + model card
uvicorn api.app:app --port 7860    # API; Swagger UI at http://localhost:7860/docs
streamlit run dashboard/app.py     # reviewer dashboard
```

Score the bundled example (in Windows PowerShell, type `curl.exe`, because `curl` is an
alias for a different command there):

```bash
curl -X POST http://localhost:7860/score -H "Content-Type: application/json" -d @sample_order.json
```

`python run.py` overwrites `models/` and `outputs/`. On the machine that produced the
committed artefacts it reproduces them exactly; on another OS or CPU the retrained model
differs slightly (section 6).

## 3. Testing

```bash
pytest tests/ -v                     # 60 tests
python -m evaluate.leakage_test      # full-size leakage suite + positive control
python -m evaluate.diagnostics       # ceiling, baselines, importance views, confidence intervals
```

| Test file | Tests | What it pins down |
|---|---|---|
| `tests/test_api.py` | 34 | API contract; `/health` 503 when the model is missing; every fail-closed path (unknown category, missing model, contradictory payment flags, missing return timing); defense-only surface (no internals endpoints, no threshold on `/health`, batch cap, reason wording); CORS |
| `tests/test_model.py` | 21 | Generator properties (latent traits never exported, no single feature separates the label, determinism); grouped split; early-stopping configuration; strictly monotone calibration and the 1.0 sentinel; adversarial slice; honest real-data states; explainer output |
| `tests/test_leakage.py` | 5 | The leakage suite on project data, and the positive control: a deliberately leaky dataset must fail it |

`python -m evaluate.leakage_test` exits with code 0 only when the project data passes all
three checks **and** the leaky control is flagged, and writes both reports to
`outputs/leakage_report.json`. CI (`.github/workflows/ci.yml`) runs the test suite and then
this command on every push to `main` and every pull request, installing from
`requirements-lock.txt` on the Python version named in `.python-version`.

## 4. Deployment (Render)

Two web services, both built from this repository's `main` branch.

| Service | Runtime | Settings |
|---|---|---|
| API | Docker (`Dockerfile`) | Health check path `/health`. The container listens on `$PORT`, which Render sets. It serves the committed `models/` artefacts; there is no training step in the image. |
| Dashboard | Python 3.13 | Build `pip install -r requirements-lock.txt`; start `streamlit run dashboard/app.py --server.address 0.0.0.0 --server.port $PORT` |

The dashboard's Python version comes from `.python-version` (`3.13`; Render picks the
latest 3.13 patch release). A `PYTHON_VERSION` environment variable on the Render service
overrides that file, so if one is set it should name a 3.13 release such as `3.13.5`. The
lock file has no wheels for newer Pythons (numpy 2.1, pyarrow 19, numba 0.61).

Render redeploys both services automatically on every push to `main` (Auto-Deploy "On
Commit"). To redeploy by hand: the service's page, **Manual Deploy → Deploy latest
commit**. A commit message containing `[skip render]` skips the redeploy, which suits
changes that touch only CI or documentation.

## 5. Verifying a deploy

1. `GET /health` returns 200 with `"model_loaded": true` and
   `"model_version": "rrs-b3ff1277c1aa"`, the hash of the committed `models/model.pkl`.
   The dashboard sidebar shows the same version.
2. `POST /score` with `sample_order.json` returns `risk_score` 0.8066, `manual_review`
   and `threshold_used` 0.0994…, matching the README.
3. On the dashboard, **Score a CSV batch → Score 25 sample return requests** scores 25
   requests, 4 of them flagged.

Free Render services sleep after 15 minutes without traffic and take about a minute to
wake, so open both URLs a few minutes before a demo. The in-process rate limiter resets
whenever the service restarts.

## 6. Reproducing the numbers

- **Same machine, lock file installed:** `python run.py` reproduces every committed number.
- **Different OS or CPU:** the generated data is identical (NumPy's generator is
  platform-independent), but XGBoost's floating-point training arithmetic is not
  bit-identical across platforms. A Linux run with seed 42 stops at 87 trees instead of
  92 and reports PR-AUC 0.4367, recall 0.766.
- **The spread to quote:** a customer-level bootstrap of the test set gives a 95% interval
  of 0.40–0.48 for PR-AUC and 52–60% for the cost reduction; five generator seeds give
  PR-AUC 0.43 ± 0.01 (`python -m evaluate.diagnostics`).
- **Cost assumptions:** the sensitivity grid shows the chosen threshold costs 10–12% more
  than optimal if the true FN:FP ratio is off by 2×, and 60–107% more if it is off by 5×.

## 7. Scope

The data is synthetic, the costs are illustrative, and the serving layer is demo-grade.
The README's [Scope and production roadmap](README.md#scope-and-production-roadmap) lists
each boundary with the next step a production deployment would take.
