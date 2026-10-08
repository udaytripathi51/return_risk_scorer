"""FastAPI service for the Return Risk Scorer.

OPERATIONAL CONTRACT
--------------------
`/health` returns 503 when the model is not loaded. A health check that always answers
200 lets an orchestrator route traffic to an instance that fails every request closed;
this one makes a broken instance visible.

CORS uses wildcard origins WITHOUT credentials. `allow_origins=["*"]` together with
`allow_credentials=True` is rejected by browsers, and the usual workaround -- reflecting
the caller's origin -- is strictly worse. This API has no cookie or session auth, so
credentials stay off and the wildcard is safe.

DEFENSE-ONLY: NO GLOBAL THRESHOLD ON /health
--------------------------------------------
Track 02 disqualifies anything offense-capable, and this service reads that strictly: no
endpoint reveals an exact decision threshold beyond what is needed to act on a single
order. `/health` is an unauthenticated liveness probe, not an act on an order, so a global
threshold there would be disclosure with no operational justification.
`/score` does return `threshold_used`, which sits squarely inside that carve-out: the
merchant needs it to interpret the one order they asked about, and it reveals nothing an
attacker could not already infer from the `risk_score` + `action` pair in the same
response. `/health` reports `threshold_configured: true` instead. See SAFETY.md.
"""
from __future__ import annotations

import logging
import os
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.schemas import (
    MAX_BATCH_SIZE,
    BatchRequest,
    BatchResponse,
    OrderRequest,
    ScoreResponse,
)
from api.service import RiskScoringService

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
logger = logging.getLogger(__name__)

service: RiskScoringService | None = None

# --- Anti-probing rate limit --------------------------------------------------------
# Defense-only: no capability may help an attacker map the decision boundary.
# The cheapest such attack is high-volume probing: vary one field, watch the score move,
# reconstruct the surface. A batch cap (api/schemas.py) plus this limiter make that slow
# and visible. Deliberately simple and in-process: this is a demo-grade control, and a
# real deployment would enforce it at the gateway with a shared store. Documented as such
# rather than overclaimed.
RATE_LIMIT_REQUESTS = int(os.environ.get("RATE_LIMIT_REQUESTS", "120"))
RATE_LIMIT_WINDOW_S = int(os.environ.get("RATE_LIMIT_WINDOW_S", "60"))
_hits: dict[str, deque[float]] = defaultdict(deque)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global service
    service = RiskScoringService()
    if not service.is_ready():
        # Not fatal: the service still answers, and every answer fails closed. /health
        # reports 503 so a deployment notices.
        logger.error("startup: model NOT loaded — all scoring will fail closed")
    yield


app = FastAPI(
    title="Return Risk Scorer",
    version="1.0.0",
    description=(
        "Defense-only return-fraud risk scoring for Indian e-commerce. Returns a "
        "calibrated risk score, a bounded action (auto_approve | manual_review) and the "
        "top-3 human-readable risk drivers. Fails closed to manual_review on any error."
    ),
    lifespan=lifespan,
)

# Wildcard origins WITHOUT credentials (see the module docstring).
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def _rate_limit(request: Request, call_next):
    if request.url.path.startswith("/score"):
        client = request.client.host if request.client else "unknown"
        now = time.monotonic()
        q = _hits[client]
        while q and now - q[0] > RATE_LIMIT_WINDOW_S:
            q.popleft()
        if len(q) >= RATE_LIMIT_REQUESTS:
            logger.warning("rate limit exceeded for %s", client)
            return JSONResponse(
                status_code=429,
                content={
                    "detail": (
                        f"Rate limit exceeded ({RATE_LIMIT_REQUESTS} scoring requests "
                        f"per {RATE_LIMIT_WINDOW_S}s). This limit exists to prevent "
                        "automated probing of the decision boundary."
                    )
                },
            )
        q.append(now)
    return await call_next(request)


@app.get("/")
async def root():
    return {
        "service": "Return Risk Scorer",
        "version": "1.0.0",
        "track": "Razorpay AI Buildathon 2026 — Track 02: AI Risk Manager",
        "posture": "defense-only, fail-closed",
        "endpoints": ["/health", "/score", "/score/batch", "/docs"],
        "max_batch_size": MAX_BATCH_SIZE,
    }


@app.get("/health")
async def health():
    """Liveness + readiness. 503 when the model is not loaded."""
    if service is None or not service.is_ready():
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "model_loaded": False,
                "detail": "Model artefacts unavailable; all scoring fails closed to manual_review.",
            },
        )
    return {
        "status": "healthy",
        "model_loaded": True,
        # Deliberately NOT the threshold value itself — see the module docstring.
        "threshold_configured": True,
        "features": len(service.features),
        "categories": service.category_names,
        "model_version": service.model_version,
    }


@app.post("/score", response_model=ScoreResponse)
async def score(order: OrderRequest):
    assert service is not None
    return service.score(order.model_dump())


@app.post("/score/batch", response_model=BatchResponse)
async def score_batch(batch: BatchRequest):
    """Batch scoring, capped at MAX_BATCH_SIZE orders per call.

    The cap is a safety control, not a performance one: unbounded batch scoring is the
    most efficient way to reverse-engineer a decision boundary.
    """
    assert service is not None
    return service.score_batch([o.model_dump() for o in batch.orders])


if __name__ == "__main__":
    import uvicorn

    # Port 7860 locally (Hugging Face Spaces' default); hosts such as Render set $PORT.
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "7860")))
