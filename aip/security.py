from __future__ import annotations

import hashlib
import json
import threading
import time
from pathlib import Path
from typing import Dict, Tuple


class RateLimiter:
    """Thread-safe token bucket limiter keyed by client identity."""

    def __init__(self, rate_per_minute: int, burst: int) -> None:
        self.rate_per_sec = max(1e-6, rate_per_minute / 60.0)
        self.burst = max(1.0, float(burst))
        self._state: Dict[str, Tuple[float, float]] = {}
        self._lock = threading.Lock()

    def allow(self, key: str, now: float | None = None) -> tuple[bool, float]:
        t = time.time() if now is None else now
        with self._lock:
            tokens, last = self._state.get(key, (self.burst, t))
            elapsed = max(0.0, t - last)
            tokens = min(self.burst, tokens + elapsed * self.rate_per_sec)
            if tokens >= 1.0:
                tokens -= 1.0
                self._state[key] = (tokens, t)
                return True, 0.0

            deficit = 1.0 - tokens
            retry_after = deficit / self.rate_per_sec
            self._state[key] = (tokens, t)
            return False, retry_after


class AuditLogger:
    def __init__(self, path: str, enabled: bool = True) -> None:
        self.enabled = enabled
        self.path = Path(path)
        self._lock = threading.Lock()

    def write(self, event: dict) -> None:
        if not self.enabled:
            return
        line = json.dumps(event, separators=(",", ":"), ensure_ascii=True)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            with self.path.open("a", encoding="utf-8") as f:
                f.write(line)
                f.write("\n")


class MetricsStore:
    def __init__(self) -> None:
        self.started_at = time.time()
        self._lock = threading.Lock()
        self.total_requests = 0
        self.total_errors = 0
        self.endpoint_counts: Dict[str, int] = {}
        self.status_counts: Dict[str, int] = {}
        self.decision_counts: Dict[str, int] = {}
        self.total_latency_ms = 0.0

    def observe(self, endpoint: str, status_code: int, latency_ms: float) -> None:
        with self._lock:
            self.total_requests += 1
            if status_code >= 400:
                self.total_errors += 1
            self.total_latency_ms += max(0.0, latency_ms)
            self.endpoint_counts[endpoint] = self.endpoint_counts.get(endpoint, 0) + 1
            s_key = str(status_code)
            self.status_counts[s_key] = self.status_counts.get(s_key, 0) + 1

    def observe_decision(self, decision: str) -> None:
        with self._lock:
            self.decision_counts[decision] = self.decision_counts.get(decision, 0) + 1

    def snapshot(self) -> dict:
        with self._lock:
            avg_latency = self.total_latency_ms / self.total_requests if self.total_requests else 0.0
            uptime_sec = max(0.0, time.time() - self.started_at)
            return {
                "uptime_seconds": round(uptime_sec, 3),
                "total_requests": self.total_requests,
                "total_errors": self.total_errors,
                "error_rate": round(self.total_errors / self.total_requests, 6) if self.total_requests else 0.0,
                "avg_latency_ms": round(avg_latency, 3),
                "endpoint_counts": dict(sorted(self.endpoint_counts.items())),
                "status_counts": dict(sorted(self.status_counts.items())),
                "decision_counts": dict(sorted(self.decision_counts.items())),
            }


def anonymize_identity(value: str) -> str:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
    return digest[:16]


def is_path_within_roots(path: Path, allowed_roots: list[str]) -> bool:
    target = path.resolve()
    for root in allowed_roots:
        try:
            root_path = Path(root).expanduser().resolve()
            target.relative_to(root_path)
            return True
        except Exception:
            continue
    return False
