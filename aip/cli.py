from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .analyzers import runtime_capabilities
from .calibration import load_calibration, save_calibration
from .config import load_settings
from .engine import analyze
from .evaluation import build_calibration_bundle, evaluate_thresholds, run_dataset, write_json_report
from .scorecard import build_readiness_scorecard, write_scorecard_files


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="aip",
        description="Authenticity Intelligence Platform",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    analyze_cmd = sub.add_parser("analyze", help="Analyze text or media authenticity risk")
    analyze_cmd.add_argument("--input", help="Path to image/audio/video/text file")
    analyze_cmd.add_argument("--text", help="Raw text input to analyze")
    analyze_cmd.add_argument(
        "--modality",
        default="auto",
        choices=["auto", "text", "image", "audio", "video", "binary"],
        help="Content modality override",
    )
    analyze_cmd.add_argument("--identity-claim", help="Claimed speaker/author identity", default=None)
    analyze_cmd.add_argument(
        "--profile",
        default="industry_low_fp",
        choices=["industry_low_fp", "balanced", "high_recall"],
        help="Decision profile. industry_low_fp is strictest against false positives.",
    )
    analyze_cmd.add_argument("--calibration", default=None, help="Calibration JSON path")
    analyze_cmd.add_argument("--output", help="Write JSON output to file", default=None)
    analyze_cmd.add_argument("--pretty", action="store_true", help="Pretty-print output JSON")

    doctor_cmd = sub.add_parser("doctor", help="Show available local open-source integrations")
    doctor_cmd.add_argument("--pretty", action="store_true", help="Pretty-print output JSON")

    preflight_cmd = sub.add_parser("preflight", help="Production readiness preflight checks")
    preflight_cmd.add_argument("--pretty", action="store_true", help="Pretty-print output JSON")

    eval_cmd = sub.add_parser("evaluate", help="Run benchmark evaluation from labeled CSV")
    eval_cmd.add_argument("--dataset", required=True, help="CSV with at least label + (input_path or text)")
    eval_cmd.add_argument(
        "--profile",
        default="industry_low_fp",
        choices=["industry_low_fp", "balanced", "high_recall"],
        help="Decision profile",
    )
    eval_cmd.add_argument("--threshold", type=float, default=0.65, help="Input threshold for metric snapshot")
    eval_cmd.add_argument("--target-fpr", type=float, default=0.03, help="FPR target for threshold recommendation")
    eval_cmd.add_argument("--output", default="evaluation_report.json", help="Evaluation JSON output")
    eval_cmd.add_argument("--scorecard-json", default="scorecard.json", help="Readiness scorecard JSON output")
    eval_cmd.add_argument("--scorecard-md", default="scorecard.md", help="Readiness scorecard markdown output")
    eval_cmd.add_argument("--calibration-output", default="calibration.json", help="Calibration JSON output")
    eval_cmd.add_argument("--pretty", action="store_true", help="Pretty-print console output")

    serve_cmd = sub.add_parser("serve", help="Run FastAPI server")
    serve_cmd.add_argument("--host", default="0.0.0.0")
    serve_cmd.add_argument("--port", type=int, default=8000)
    serve_cmd.add_argument("--workers", type=int, default=1)
    serve_cmd.add_argument("--log-level", default="info", choices=["critical", "error", "warning", "info", "debug"])
    serve_cmd.add_argument("--reload", action="store_true")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "analyze":
        if not args.input and args.text is None:
            parser.error("analyze requires either --input or --text")

        result = analyze(
            input_path=args.input,
            text=args.text,
            modality=args.modality,
            identity_claim=args.identity_claim,
            policy_profile=args.profile,
            calibration=load_calibration(args.calibration),
        )

        json_output = json.dumps(result.to_dict(), indent=2 if args.pretty else None)

        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(json_output)
            print(f"Wrote analysis to {args.output}")
        else:
            print(json_output)

    elif args.command == "doctor":
        payload = {
            "runtime_capabilities": runtime_capabilities(),
            "notes": [
                "All integrations are local and optional.",
                "No account-based APIs are required by this tool.",
                "Missing components are auto-skipped with lower confidence coverage.",
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
                "name": "path_restrictions_configured",
                "pass": len(cfg.allowed_input_roots) > 0,
                "detail": f"allowed_input_roots={cfg.allowed_input_roots}",
            },
            {
                "name": "video_tooling_available",
                "pass": bool(caps.get("ffprobe")) and bool(caps.get("ffmpeg")),
                "detail": f"ffprobe={caps.get('ffprobe')}, ffmpeg={caps.get('ffmpeg')}",
            },
            {
                "name": "forensics_stack_available",
                "pass": bool(caps.get("numpy")) and bool(caps.get("pillow")) and bool(caps.get("scipy_wav")),
                "detail": f"numpy={caps.get('numpy')}, pillow={caps.get('pillow')}, scipy_wav={caps.get('scipy_wav')}",
            },
            {
                "name": "calibration_configured_or_optional",
                "pass": (not cfg.calibration_file) or Path(cfg.calibration_file).expanduser().exists(),
                "detail": (
                    f"calibration_file={cfg.calibration_file}"
                    if cfg.calibration_file
                    else "no calibration configured (recommended after benchmark tuning)"
                ),
            },
        ]

        passed = sum(1 for c in checks if c["pass"])
        total = len(checks)
        payload = {
            "summary": {
                "passed": passed,
                "total": total,
                "status": "ready" if passed == total else "needs_hardening",
            },
            "checks": checks,
            "notes": [
                "Market launch should wait until all critical checks pass and benchmark scorecard gates pass.",
            ],
        }
        print(json.dumps(payload, indent=2 if args.pretty else None))

    elif args.command == "evaluate":
        raw = run_dataset(dataset_csv=args.dataset, profile=args.profile)
        report = evaluate_thresholds(
            raw_results=raw,
            threshold=args.threshold,
            target_fpr=args.target_fpr,
        )
        write_json_report(report, args.output)
        calibration_bundle = build_calibration_bundle(report)
        calibration_path = save_calibration(calibration_bundle, args.calibration_output)

        scorecard = build_readiness_scorecard(report)
        scorecard_json_path, scorecard_md_path = write_scorecard_files(
            report,
            scorecard,
            json_path=args.scorecard_json,
            markdown_path=args.scorecard_md,
        )

        console = {
            "evaluation_report": args.output,
            "scorecard_json": scorecard_json_path,
            "scorecard_md": scorecard_md_path,
            "calibration_json": calibration_path,
            "readiness_score": scorecard.get("readiness_score"),
            "readiness_band": scorecard.get("readiness_band"),
            "blocked_by": scorecard.get("blocked_by", []),
            "recommended_threshold": report.get("recommended_threshold", {}),
        }
        print(json.dumps(console, indent=2 if args.pretty else None))

    elif args.command == "serve":
        try:
            import uvicorn
        except Exception as exc:
            raise RuntimeError("uvicorn is not installed. Install with: pip install -e '.[api]'") from exc

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
