from __future__ import annotations

import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .analyzers import runtime_capabilities
from .calibration import load_calibration
from .config import Settings, load_settings
from .engine import analyze
from .security import AuditLogger, MetricsStore, RateLimiter, anonymize_identity, is_path_within_roots

try:
    from fastapi import Depends, FastAPI, Header, HTTPException, Request
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel, Field
except Exception as exc:  # pragma: no cover - only imported during runtime
    raise RuntimeError(
        "FastAPI dependencies are not installed. Install with: pip install -e '.[api]'"
    ) from exc


class AnalyzeRequest(BaseModel):
    input_path: Optional[str] = Field(default=None, description="Path to local asset")
    text: Optional[str] = Field(default=None, description="Inline text to analyze")
    modality: str = Field(default="auto")
    identity_claim: Optional[str] = Field(default=None)
    profile: str = Field(default="industry_low_fp")


class AnalyzeResponse(BaseModel):
    request_id: str
    processed_at: str
    result: dict


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
    calibration = load_calibration(cfg.calibration_file)
    limiter = RateLimiter(cfg.rate_limit_per_minute, cfg.rate_limit_burst)
    metrics = MetricsStore()
    audit_logger = AuditLogger(path=cfg.audit_log_path, enabled=cfg.enable_audit_log)
    started_at = _utc_now()

    app = FastAPI(title=cfg.app_name, version=cfg.app_version)
    app.state.cfg = cfg
    app.state.metrics = metrics
    app.state.limiter = limiter
    app.state.audit = audit_logger
    app.state.calibration = calibration

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
                headers={"Retry-After": str(max(1, int(retry_after)))}
            )

        return {
            "client_hash": anonymize_identity(client_seed),
            "auth": bool(cfg.api_key),
        }

    @app.get("/healthz")
    def healthz() -> dict:
        return {
            "status": "ok",
            "service": "aip",
            "version": cfg.app_version,
            "started_at": started_at,
            "time_utc": _utc_now(),
        }

    @app.get("/readyz")
    def readyz() -> dict:
        caps = runtime_capabilities()
        degraded_reasons = []
        if not caps.get("numpy") or not caps.get("pillow"):
            degraded_reasons.append("image_forensics_degraded")
        if not caps.get("ffprobe"):
            degraded_reasons.append("video_probe_degraded")
        if not caps.get("ffmpeg"):
            degraded_reasons.append("video_frame_forensics_degraded")

        status = "ready_degraded" if degraded_reasons else "ready"
        return {
            "status": status,
            "service": "aip",
            "version": cfg.app_version,
            "degraded_reasons": degraded_reasons,
            "runtime_capabilities": caps,
        }

    @app.get("/doctor")
    def doctor(_: dict = Depends(_auth_and_limit)) -> dict:
        return {
            "runtime_capabilities": runtime_capabilities(),
            "notes": [
                "No account-based third-party API required.",
                "Missing optional integrations lower observability instead of hard failure.",
                "Guardrails enabled: api-key, rate-limit, audit log, path restrictions.",
            ],
        }

    @app.get("/policies")
    def policies(_: dict = Depends(_auth_and_limit)) -> dict:
        return {
            "max_text_chars": cfg.max_text_chars,
            "max_file_size_mb": cfg.max_file_size_mb,
            "rate_limit_per_minute": cfg.rate_limit_per_minute,
            "rate_limit_burst": cfg.rate_limit_burst,
            "allowed_input_roots": cfg.allowed_input_roots,
            "audit_log_enabled": cfg.enable_audit_log,
            "metrics_enabled": cfg.enable_metrics,
            "api_key_required": bool(cfg.api_key),
            "calibration_file": cfg.calibration_file or None,
            "calibration_loaded": bool(calibration),
        }

    @app.get("/calibration")
    def calibration_info(_: dict = Depends(_auth_and_limit)) -> dict:
        if not calibration:
            return {"loaded": False}
        return {
            "loaded": True,
            "profiles": sorted(list((calibration.get("profiles") or {}).keys())) if isinstance(calibration, dict) else [],
            "default_threshold": calibration.get("default_threshold") if isinstance(calibration, dict) else None,
        }

    @app.get("/metrics")
    def metrics_endpoint(_: dict = Depends(_auth_and_limit)) -> dict:
        if not cfg.enable_metrics:
            return {"enabled": False}
        return {
            "enabled": True,
            "snapshot": metrics.snapshot(),
        }

    @app.post("/analyze", response_model=AnalyzeResponse)
    def analyze_endpoint(payload: AnalyzeRequest, request: Request, ctx: dict = Depends(_auth_and_limit)) -> AnalyzeResponse:
        request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
        started = time.perf_counter()
        event = {
            "time_utc": _utc_now(),
            "request_id": request_id,
            "endpoint": "/analyze",
            "client_hash": ctx["client_hash"],
            "profile": payload.profile,
            "modality": payload.modality,
            "status_code": 200,
        }

        try:
            if not payload.input_path and not payload.text:
                raise HTTPException(status_code=400, detail="Either input_path or text must be provided")

            if payload.text and len(payload.text) > cfg.max_text_chars:
                raise HTTPException(
                    status_code=413,
                    detail=f"Text exceeds max length ({cfg.max_text_chars} chars)",
                )

            if payload.input_path:
                path = Path(payload.input_path).expanduser().resolve()
                if not is_path_within_roots(path, cfg.allowed_input_roots):
                    raise HTTPException(status_code=403, detail="input_path is outside allowed roots")
                if not path.exists():
                    raise HTTPException(status_code=404, detail="input_path not found")

                max_size_bytes = cfg.max_file_size_mb * 1024 * 1024
                size = path.stat().st_size
                if size > max_size_bytes:
                    raise HTTPException(
                        status_code=413,
                        detail=f"input_path exceeds max file size ({cfg.max_file_size_mb} MB)",
                    )

            result = analyze(
                input_path=payload.input_path,
                text=payload.text,
                modality=payload.modality,
                identity_claim=payload.identity_claim,
                policy_profile=payload.profile,
                calibration=calibration,
            ).to_dict()

            decision = str(result.get("risk", {}).get("decision", "unknown"))
            if cfg.enable_metrics:
                metrics.observe_decision(decision)

            event.update(
                {
                    "status_code": 200,
                    "decision": decision,
                    "overall_risk": result.get("risk", {}).get("overall_risk"),
                    "confidence": result.get("risk", {}).get("confidence"),
                    "uncertainty": result.get("risk", {}).get("uncertainty"),
                    "coverage": result.get("detection", {}).get("coverage"),
                    "quality": result.get("detection", {}).get("quality"),
                    "provenance_verified": result.get("provenance", {}).get("verified"),
                }
            )

            return AnalyzeResponse(
                request_id=request_id,
                processed_at=_utc_now(),
                result=result,
            )

        except HTTPException as exc:
            event.update({"status_code": exc.status_code, "error": str(exc.detail)})
            raise
        except FileNotFoundError:
            event.update({"status_code": 404, "error": "input_path not found"})
            raise HTTPException(status_code=404, detail="input_path not found")
        except Exception:
            event.update({"status_code": 500, "error": "internal_error"})
            raise HTTPException(status_code=500, detail="Internal server error")
        finally:
            event["latency_ms"] = round((time.perf_counter() - started) * 1000.0, 3)
            audit_logger.write(event)

    return app


app = create_app()
