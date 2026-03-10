"""
model_server.py
===============
MCP (Model Classification Protocol) Server — a standalone FastAPI service
that hosts all 3 trained models and exposes classification endpoints.

Run separately from the Gmail poller so models stay warm in GPU memory.

Endpoints:
    POST /classify          — classify a single email
    POST /classify/batch    — classify multiple emails
    GET  /health            — check model status
    GET  /models            — list loaded models + configs

Usage:
    CUDA_VISIBLE_DEVICES=0 uvicorn jenkins.model_server:app --host 0.0.0.0 --port 8001

    # Or with 2 GPUs (classifier on 0, gemma on 1):
    uvicorn jenkins.model_server:app --host 0.0.0.0 --port 8001
"""

import time
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Will be set during lifespan
classifier = None


# ── Request / Response schemas ────────────────────────────────────────────────
class ClassifyRequest(BaseModel):
    email_text: str = Field(..., description="Full email text (subject + body)")
    force_deep: bool = Field(False, description="Force deep Gemma analysis regardless of gate")

class ClassifyResponse(BaseModel):
    label: str
    confidence: float
    reason: str = ""
    indicators: list = []
    action: str = ""
    path: str = "fast"
    latency_ms: float = 0.0

class BatchClassifyRequest(BaseModel):
    emails: list[ClassifyRequest]

class BatchClassifyResponse(BaseModel):
    results: list[ClassifyResponse]
    total_latency_ms: float = 0.0

class ModelInfo(BaseModel):
    name: str
    status: str
    device: str
    details: dict = {}


# ── App lifecycle ─────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global classifier
    print("=" * 60)
    print("🚀 MCP Model Server — Loading all models …")
    print("=" * 60)

    from jenkins.orchestrator import EmailClassifier
    classifier = EmailClassifier(
        classifier_device="cuda:0",
        gemma_device="cuda:0",
    )

    print("=" * 60)
    print("✅ MCP Server ready! All models loaded.")
    print("=" * 60)
    yield
    print("Shutting down MCP Server …")


app = FastAPI(
    title="Email Threat MCP Server",
    description="Model Classification Protocol server hosting RoBERTa classifier, gate, and Gemma LoRA",
    version="1.0.0",
    lifespan=lifespan,
)


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status": "healthy" if classifier else "loading",
        "models_loaded": classifier is not None,
    }


@app.get("/models")
def list_models():
    if not classifier:
        return {"status": "loading"}
    return {
        "models": [
            ModelInfo(
                name="roberta-classifier",
                status="loaded",
                device=str(classifier.classifier_device),
                details={
                    "labels": list(classifier.id2label.values()),
                    "max_length": 512,
                }
            ).model_dump(),
            ModelInfo(
                name="open-set-gate",
                status="loaded",
                device="cpu",
                details={
                    "temperature": classifier.temperature,
                    "threshold": classifier.threshold,
                }
            ).model_dump(),
            ModelInfo(
                name="gemma-2b-lora",
                status="loaded",
                device=str(classifier.gemma_device),
                details={
                    "base_model": "google/gemma-2b-it",
                    "quantization": "4-bit NF4",
                }
            ).model_dump(),
        ]
    }


@app.post("/classify", response_model=ClassifyResponse)
async def classify_email(req: ClassifyRequest):
    """Classify a single email through the 3-component pipeline."""
    if not classifier:
        raise HTTPException(503, "Models still loading")

    t0 = time.perf_counter()

    if req.force_deep:
        # Bypass gate, go straight to Gemma
        result = classifier._classify_deep(req.email_text)
        result["path"] = "deep (forced)"
    else:
        result = classifier.classify(req.email_text)

    latency = (time.perf_counter() - t0) * 1000

    return ClassifyResponse(
        label=result.get("label", "unknown"),
        confidence=result.get("confidence", 0.0),
        reason=result.get("reason", ""),
        indicators=result.get("indicators", []),
        action=result.get("action", "review"),
        path=result.get("path", "fast"),
        latency_ms=round(latency, 1),
    )


@app.post("/classify/batch", response_model=BatchClassifyResponse)
async def classify_batch(req: BatchClassifyRequest):
    """Classify multiple emails."""
    if not classifier:
        raise HTTPException(503, "Models still loading")

    t0 = time.perf_counter()
    results = []
    for email_req in req.emails:
        t1 = time.perf_counter()
        if email_req.force_deep:
            raw = classifier._classify_deep(email_req.email_text)
            raw["path"] = "deep (forced)"
        else:
            raw = classifier.classify(email_req.email_text)
        lat = (time.perf_counter() - t1) * 1000

        results.append(ClassifyResponse(
            label=raw.get("label", "unknown"),
            confidence=raw.get("confidence", 0.0),
            reason=raw.get("reason", ""),
            indicators=raw.get("indicators", []),
            action=raw.get("action", "review"),
            path=raw.get("path", "fast"),
            latency_ms=round(lat, 1),
        ))

    total = (time.perf_counter() - t0) * 1000
    return BatchClassifyResponse(results=results, total_latency_ms=round(total, 1))
