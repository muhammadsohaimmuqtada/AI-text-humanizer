"""FastAPI application for the AI Text Humanizer Platform."""

from __future__ import annotations

import time
import uuid
from datetime import datetime, timezone
from typing import Optional

from .analyzers import runtime_capabilities
from .config import Settings, load_settings
from .humanizer import HumanizeResult, humanize
from .security import AuditLogger, MetricsStore, RateLimiter, anonymize_identity

try:
    from fastapi import Depends, FastAPI, Header, HTTPException, Request
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel, Field
except Exception as exc:  # pragma: no cover - only imported during runtime
    raise RuntimeError(
        "FastAPI dependencies are not installed. Install with: pip install -e '.[api]'"
    ) from exc


class HumanizeRequest(BaseModel):
    text: str = Field(..., description="AI-generated text to humanize")
    merge_rate: float = Field(default=0.25, ge=0.0, le=1.0, description="Probability of merging adjacent sentences")
    seed: Optional[int] = Field(default=None, description="Optional random seed for reproducible output")
    adversarial_mode: bool = Field(default=False, description="Enable zero-width space injection and homoglyph swapping")


class HumanizeResponse(BaseModel):
    request_id: str
    processed_at: str
    humanized_text: str
    original_word_count: int
    humanized_word_count: int
    markers_removed: int
    sentences_merged: int


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _client_seed(request: Request, x_api_key: Optional[str]) -> str:
    if x_api_key:
        return f"api:{x_api_key}"
    xff = request.headers.get("x-forwarded-for", "").strip()
    if xff:
        return f"ip:{xff.split(',')[0].strip()}"
    host = request.client.host if request.client else "unknown"
    return f"ip:{host}"


def create_app(settings: Settings | None = None) -> FastAPI:
    cfg = settings or load_settings()
    limiter = RateLimiter(cfg.rate_limit_per_minute, cfg.rate_limit_burst)
    metrics = MetricsStore()
    audit_logger = AuditLogger(path=cfg.audit_log_path, enabled=cfg.enable_audit_log)
    started_at = _utc_now()

    app = FastAPI(title=cfg.app_name, version=cfg.app_version)
    app.state.cfg = cfg
    app.state.metrics = metrics
    app.state.limiter = limiter
    app.state.audit = audit_logger

    if cfg.enable_cors:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=cfg.cors_origins,
            allow_credentials=False,
            allow_methods=["GET", "POST"],
            allow_headers=["*"],
        )

    @app.middleware("http")
    async def request_context_middleware(request: Request, call_next):
        request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
        request.state.request_id = request_id
        start = time.perf_counter()
        try:
            response = await call_next(request)
        except HTTPException as exc:
            response = JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})
        except Exception:
            response = JSONResponse(status_code=500, content={"detail": "Internal server error"})

        latency_ms = (time.perf_counter() - start) * 1000.0
        if cfg.enable_metrics:
            metrics.observe(request.url.path, response.status_code, latency_ms)
        response.headers["x-request-id"] = request_id
        response.headers["x-process-time-ms"] = f"{latency_ms:.3f}"
        return response

    def _auth_and_limit(
        request: Request,
        x_api_key: Optional[str] = Header(default=None),
    ) -> dict:
        if cfg.api_key and x_api_key != cfg.api_key:
            raise HTTPException(status_code=401, detail="Invalid API key")

        client_seed = _client_seed(request, x_api_key)
        allowed, retry_after = limiter.allow(client_seed)
        if not allowed:
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded",
                headers={"Retry-After": str(max(1, int(retry_after)))},
            )

        return {
            "client_hash": anonymize_identity(client_seed),
            "auth": bool(cfg.api_key),
        }

    @app.get("/healthz")
    def healthz() -> dict:
        return {
            "status": "ok",
            "service": "ai-text-humanizer",
            "version": cfg.app_version,
            "started_at": started_at,
            "time_utc": _utc_now(),
        }

    @app.get("/readyz")
    def readyz() -> dict:
        return {
            "status": "ready",
            "service": "ai-text-humanizer",
            "version": cfg.app_version,
            "degraded_reasons": [],
        }

    @app.get("/doctor")
    def doctor(_: dict = Depends(_auth_and_limit)) -> dict:
        return {
            "runtime_capabilities": runtime_capabilities(),
            "notes": [
                "All NLP processing is performed locally; no external API or download required.",
                "The evasion engine uses zero-width spaces, homoglyph swapping, contraction",
                "manipulation, and burstiness variation — no NLTK or WordNet dependency.",
            ],
        }

    @app.get("/policies")
    def policies(_: dict = Depends(_auth_and_limit)) -> dict:
        return {
            "max_text_chars": cfg.max_text_chars,
            "rate_limit_per_minute": cfg.rate_limit_per_minute,
            "rate_limit_burst": cfg.rate_limit_burst,
            "audit_log_enabled": cfg.enable_audit_log,
            "metrics_enabled": cfg.enable_metrics,
            "api_key_required": bool(cfg.api_key),
        }

    @app.get("/metrics")
    def metrics_endpoint(_: dict = Depends(_auth_and_limit)) -> dict:
        if not cfg.enable_metrics:
            return {"enabled": False}
        return {
            "enabled": True,
            "snapshot": metrics.snapshot(),
        }

    @app.post("/humanize", response_model=HumanizeResponse)
    def humanize_endpoint(
        payload: HumanizeRequest,
        request: Request,
        ctx: dict = Depends(_auth_and_limit),
    ) -> HumanizeResponse:
        request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
        started = time.perf_counter()
        event = {
            "time_utc": _utc_now(),
            "request_id": request_id,
            "endpoint": "/humanize",
            "client_hash": ctx["client_hash"],
            "status_code": 200,
        }

        try:
            if len(payload.text) > cfg.max_text_chars:
                raise HTTPException(
                    status_code=413,
                    detail=f"Text exceeds max length ({cfg.max_text_chars} chars)",
                )

            result: HumanizeResult = humanize(
                text=payload.text,
                merge_rate=payload.merge_rate,
                seed=payload.seed,
                adversarial_mode=payload.adversarial_mode,
            )

            event.update(
                {
                    "status_code": 200,
                    "original_word_count": result.original_word_count,
                    "humanized_word_count": result.humanized_word_count,
                    "markers_removed": result.markers_removed,
                    "sentences_merged": result.sentences_merged,
                }
            )

            return HumanizeResponse(
                request_id=request_id,
                processed_at=_utc_now(),
                humanized_text=result.humanized_text,
                original_word_count=result.original_word_count,
                humanized_word_count=result.humanized_word_count,
                markers_removed=result.markers_removed,
                sentences_merged=result.sentences_merged,
            )

        except HTTPException as exc:
            event.update({"status_code": exc.status_code, "error": str(exc.detail)})
            raise
        except Exception:
            event.update({"status_code": 500, "error": "internal_error"})
            raise HTTPException(status_code=500, detail="Internal server error")
        finally:
            event["latency_ms"] = round((time.perf_counter() - started) * 1000.0, 3)
            audit_logger.write(event)

    return app


app = create_app()
