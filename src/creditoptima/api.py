from __future__ import annotations

import json
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .config import METRICS_PATH, MODEL_PATH, ROOT
from .model import RiskEngine, probability_to_score, risk_policy
from .schemas import Applicant, HealthResponse, ScoreResponse


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_path = Path(os.getenv("CREDITOPTIMA_MODEL_PATH", MODEL_PATH))
    app.state.engine = RiskEngine.load(model_path) if model_path.exists() else None
    yield


app = FastAPI(
    title="CreditOptima API", version="1.0.0",
    description="Explainable, cost-sensitive credit default risk scoring.",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["GET", "POST"], allow_headers=["*"],
)
frontend = ROOT / "frontend"
if frontend.exists():
    app.mount("/static", StaticFiles(directory=frontend), name="static")


@app.get("/", include_in_schema=False)
def dashboard():
    return FileResponse(frontend / "index.html")


@app.get("/health", response_model=HealthResponse)
def health(request: Request):
    engine = request.app.state.engine
    return {"status": "ok" if engine else "degraded", "model_loaded": bool(engine),
            "model_version": engine.version if engine else None}


@app.get("/v1/model/metrics")
def metrics():
    if not METRICS_PATH.exists():
        raise HTTPException(503, "Metrics not available; run training first")
    return json.loads(METRICS_PATH.read_text(encoding="utf-8"))


@app.post("/v1/score", response_model=ScoreResponse)
def score(applicant: Applicant, request: Request):
    started = time.perf_counter()
    engine = request.app.state.engine
    if engine is None:
        raise HTTPException(503, "Model not available; run training first")
    probability, reasons = engine.predict(applicant.feature_dict())
    tier, decision = risk_policy(probability)
    return {
        "request_id": str(uuid4()), "default_probability": round(probability, 6),
        "credit_score": probability_to_score(probability), "risk_tier": tier,
        "decision": decision, "adverse_action_reasons": reasons,
        "model_version": engine.version,
        "latency_ms": round((time.perf_counter() - started) * 1000, 3),
    }

