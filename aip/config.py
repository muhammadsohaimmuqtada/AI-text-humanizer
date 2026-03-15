from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int, min_value: int | None = None) -> int:
    raw = os.getenv(name)
    if raw is None:
        value = default
    else:
        try:
            value = int(raw)
        except ValueError:
            value = default
    if min_value is not None:
        value = max(min_value, value)
    return value


def _env_list(name: str, default: List[str]) -> List[str]:
    raw = os.getenv(name)
    if raw is None:
        return list(default)
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    return parts if parts else list(default)


@dataclass(frozen=True)
class Settings:
    app_name: str
    app_version: str
    api_key: str
    max_text_chars: int
    max_file_size_mb: int
    rate_limit_per_minute: int
    rate_limit_burst: int
    allowed_input_roots: List[str]
    enable_audit_log: bool
    audit_log_path: str
    enable_metrics: bool
    enable_cors: bool
    cors_origins: List[str]
    calibration_file: str



def load_settings() -> Settings:
    # Secure-by-default: avoid implicit broad local roots.
    # Operators should explicitly set AIP_ALLOWED_INPUT_ROOTS in production.
    default_roots = ["/tmp"]

    return Settings(
        app_name="AI Text Humanizer API",
        app_version="1.0.0",
        api_key=os.getenv("AIP_API_KEY", "").strip(),
        max_text_chars=_env_int("AIP_MAX_TEXT_CHARS", 25000, min_value=128),
        max_file_size_mb=_env_int("AIP_MAX_FILE_SIZE_MB", 300, min_value=1),
        rate_limit_per_minute=_env_int("AIP_RATE_LIMIT_PER_MIN", 60, min_value=1),
        rate_limit_burst=_env_int("AIP_RATE_LIMIT_BURST", 20, min_value=1),
        allowed_input_roots=_env_list("AIP_ALLOWED_INPUT_ROOTS", default_roots),
        enable_audit_log=_env_bool("AIP_ENABLE_AUDIT_LOG", True),
        audit_log_path=os.getenv("AIP_AUDIT_LOG_PATH", "/tmp/aip_audit.jsonl"),
        enable_metrics=_env_bool("AIP_ENABLE_METRICS", True),
        enable_cors=_env_bool("AIP_ENABLE_CORS", False),
        cors_origins=_env_list("AIP_CORS_ORIGINS", ["*"]),
        calibration_file=os.getenv("AIP_CALIBRATION_FILE", "").strip(),
    )
