from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

from aip.api import create_app
from aip.config import load_settings
from aip.security import AuditLogger, MetricsStore, RateLimiter, anonymize_identity, is_path_within_roots


class SecurityModuleTests(unittest.TestCase):
    def test_rate_limiter_blocks_after_burst(self) -> None:
        limiter = RateLimiter(rate_per_minute=1, burst=1)
        ok1, retry1 = limiter.allow("client-a", now=100.0)
        ok2, retry2 = limiter.allow("client-a", now=100.1)
        self.assertTrue(ok1)
        self.assertFalse(ok2)
        self.assertEqual(retry1, 0.0)
        self.assertGreater(retry2, 0.0)

    def test_path_restriction_check(self) -> None:
        allowed = ["/tmp", "/srv/aip-data"]
        self.assertTrue(is_path_within_roots(Path("/tmp/demo.txt"), allowed))
        self.assertFalse(is_path_within_roots(Path("/etc/hosts"), allowed))

    def test_metrics_store_observe(self) -> None:
        m = MetricsStore()
        m.observe("/analyze", 200, 10.5)
        m.observe("/analyze", 429, 2.0)
        m.observe_decision("manual_review")
        snap = m.snapshot()
        self.assertEqual(snap["total_requests"], 2)
        self.assertEqual(snap["total_errors"], 1)
        self.assertIn("manual_review", snap["decision_counts"])

    def test_audit_logger_writes_jsonl(self) -> None:
        with tempfile.TemporaryDirectory(prefix="aip_audit_test_") as td:
            p = Path(td) / "audit.jsonl"
            logger = AuditLogger(str(p), enabled=True)
            logger.write({"event": "test", "value": 1})
            raw = p.read_text(encoding="utf-8").strip()
            self.assertTrue(raw)
            obj = json.loads(raw)
            self.assertEqual(obj["event"], "test")

    def test_anonymize_identity_stable(self) -> None:
        a = anonymize_identity("client")
        b = anonymize_identity("client")
        c = anonymize_identity("client-2")
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)


class ApiFactoryTests(unittest.TestCase):
    def test_create_app_registers_routes(self) -> None:
        app = create_app()
        paths = {route.path for route in app.routes}
        self.assertIn("/healthz", paths)
        self.assertIn("/readyz", paths)
        self.assertIn("/humanize", paths)
        self.assertIn("/metrics", paths)
        self.assertIn("/policies", paths)

    def test_load_settings_env_override(self) -> None:
        old = os.environ.get("AIP_MAX_TEXT_CHARS")
        try:
            os.environ["AIP_MAX_TEXT_CHARS"] = "777"
            cfg = load_settings()
            self.assertEqual(cfg.max_text_chars, 777)
        finally:
            if old is None:
                os.environ.pop("AIP_MAX_TEXT_CHARS", None)
            else:
                os.environ["AIP_MAX_TEXT_CHARS"] = old


if __name__ == "__main__":
    unittest.main()
