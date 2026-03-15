"""Command-line interface for the AI Text Humanizer."""

from __future__ import annotations

import argparse
import json
import sys

from .analyzers import runtime_capabilities
from .config import load_settings
from .humanizer import humanize


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="aip",
        description="AI Text Humanizer",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    humanize_cmd = sub.add_parser("humanize", help="Rewrite AI-generated text to read as human-written")
    humanize_cmd.add_argument("--text", required=True, help="Raw text to humanize")
    humanize_cmd.add_argument(
        "--synonym-rate",
        type=float,
        default=0.35,
        metavar="RATE",
        help="Fraction of adjectives/adverbs to swap via WordNet (0–1, default 0.35)",
    )
    humanize_cmd.add_argument(
        "--merge-rate",
        type=float,
        default=0.25,
        metavar="RATE",
        help="Probability of merging two adjacent sentences (0–1, default 0.25)",
    )
    humanize_cmd.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed for reproducible output",
    )
    humanize_cmd.add_argument("--output", help="Write JSON output to file", default=None)
    humanize_cmd.add_argument("--pretty", action="store_true", help="Pretty-print output JSON")

    doctor_cmd = sub.add_parser("doctor", help="Show available NLP integrations")
    doctor_cmd.add_argument("--pretty", action="store_true", help="Pretty-print output JSON")

    preflight_cmd = sub.add_parser("preflight", help="Production readiness preflight checks")
    preflight_cmd.add_argument("--pretty", action="store_true", help="Pretty-print output JSON")

    serve_cmd = sub.add_parser("serve", help="Run FastAPI server")
    serve_cmd.add_argument("--host", default="0.0.0.0")
    serve_cmd.add_argument("--port", type=int, default=8000)
    serve_cmd.add_argument("--workers", type=int, default=1)
    serve_cmd.add_argument(
        "--log-level",
        default="info",
        choices=["critical", "error", "warning", "info", "debug"],
    )
    serve_cmd.add_argument("--reload", action="store_true")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "humanize":
        result = humanize(
            text=args.text,
            synonym_rate=args.synonym_rate,
            merge_rate=args.merge_rate,
            seed=args.seed,
        )
        payload = {
            "humanized_text": result.humanized_text,
            "original_word_count": result.original_word_count,
            "humanized_word_count": result.humanized_word_count,
            "markers_removed": result.markers_removed,
            "sentences_merged": result.sentences_merged,
        }
        json_output = json.dumps(payload, indent=2 if args.pretty else None)

        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(json_output)
            print(f"Wrote humanized output to {args.output}")
        else:
            print(json_output)

    elif args.command == "doctor":
        payload = {
            "runtime_capabilities": runtime_capabilities(),
            "notes": [
                "All NLP processing is local and requires no external accounts.",
                "NLTK is used for POS tagging and WordNet synonym substitution.",
                "Run: python -m nltk.downloader wordnet punkt_tab averaged_perceptron_tagger_eng",
            ],
        }
        print(json.dumps(payload, indent=2 if args.pretty else None))

    elif args.command == "preflight":
        cfg = load_settings()
        caps = runtime_capabilities()
        checks = [
            {
                "name": "api_key_configured",
                "pass": bool(cfg.api_key),
                "detail": "Set AIP_API_KEY for non-private environments.",
            },
            {
                "name": "audit_log_enabled",
                "pass": bool(cfg.enable_audit_log),
                "detail": f"audit_log_path={cfg.audit_log_path}",
            },
            {
                "name": "rate_limit_sane",
                "pass": cfg.rate_limit_per_minute <= 300 and cfg.rate_limit_burst <= 100,
                "detail": f"rate_per_min={cfg.rate_limit_per_minute}, burst={cfg.rate_limit_burst}",
            },
            {
                "name": "nltk_available",
                "pass": bool(caps.get("nltk")),
                "detail": "Install NLTK: pip install nltk",
            },
            {
                "name": "wordnet_available",
                "pass": bool(caps.get("wordnet")),
                "detail": "Run: python -m nltk.downloader wordnet",
            },
            {
                "name": "punkt_available",
                "pass": bool(caps.get("punkt")),
                "detail": "Run: python -m nltk.downloader punkt_tab",
            },
        ]

        passed = sum(1 for c in checks if c["pass"])
        total = len(checks)
        payload = {
            "summary": {
                "passed": passed,
                "total": total,
                "status": "ready" if passed == total else "needs_setup",
            },
            "checks": checks,
            "notes": [
                "All NLP dependencies must pass for full humanization quality.",
                "Fallback regex processing is available when NLTK data is missing.",
            ],
        }
        print(json.dumps(payload, indent=2 if args.pretty else None))

    elif args.command == "serve":
        try:
            import uvicorn
        except Exception as exc:
            raise RuntimeError(
                "uvicorn is not installed. Install with: pip install -e '.[api]'"
            ) from exc

        uvicorn.run(
            "aip.api:app",
            host=args.host,
            port=args.port,
            workers=args.workers,
            log_level=args.log_level,
            reload=args.reload,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
