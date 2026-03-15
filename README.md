# Authenticity Intelligence Platform

A production-oriented authenticity system for `video`, `image`, `audio`, and `text` that combines:
- forensic detection signals
- provenance checks
- calibrated risk scoring
- workflow decisions (`allow`, `review`, `block`, `inconclusive`)
- benchmark evaluation and market-readiness scorecard
- hardened API controls (auth, rate limit, audit log, path and size guardrails)

## Project Standards

- License: MIT ([LICENSE](LICENSE))
- Security guidance: [SECURITY.md](SECURITY.md)
- Deployment guide: [DEPLOYMENT.md](DEPLOYMENT.md)
- Market readiness policy: [MARKET_READINESS.md](MARKET_READINESS.md)
- Contribution guide: [CONTRIBUTING.md](CONTRIBUTING.md)
- Code of conduct: [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md)
- Changelog: [CHANGELOG.md](CHANGELOG.md)

## Features

- Multi-family evidence consensus to reduce false positives
- Observability gating with `coverage` and `quality`
- Provenance-aware risk damping
- Decision profiles:
  - `industry_low_fp`
  - `balanced`
  - `high_recall`
- API hardening:
  - API key support
  - Token-bucket rate limiting
  - Per-request audit trail (JSONL)
  - Input path allow-list restrictions
  - Max text/file size policies
  - Request ID and latency headers
  - Health, readiness, policy, and metrics endpoints

## Install

```bash
cd /path/to/authenticity_tool
python3 -m pip install -e '.[forensics,api]'
```

## Environment configuration

Copy and edit `.env.example`:

```bash
cp .env.example .env
```

Important variables:
- `AIP_API_KEY`
- `AIP_ALLOWED_INPUT_ROOTS`
- `AIP_MAX_FILE_SIZE_MB`
- `AIP_RATE_LIMIT_PER_MIN`
- `AIP_RATE_LIMIT_BURST`
- `AIP_AUDIT_LOG_PATH`
- `AIP_CALIBRATION_FILE`

## CLI usage

Dependency health:

```bash
python3 -m aip.cli doctor --pretty
```

Production preflight:

```bash
python3 -m aip.cli preflight --pretty
```

Analyze media or text:

```bash
python3 -m aip.cli analyze --input /path/to/file.jpg --modality auto --profile industry_low_fp --pretty
python3 -m aip.cli analyze --text "I am the CEO, wire funds now." --identity-claim "CEO Name" --profile industry_low_fp --pretty
python3 -m aip.cli analyze --input /path/to/file.mp4 --profile industry_low_fp --calibration calibration.json --pretty
```

## API usage

Run service:

```bash
python3 -m aip.cli serve --host 0.0.0.0 --port 8000 --workers 2 --log-level info
```

Health and readiness:

```bash
curl http://127.0.0.1:8000/healthz
curl http://127.0.0.1:8000/readyz
```

Policy and metrics (with API key if set):

```bash
curl -H 'x-api-key: your-secret' http://127.0.0.1:8000/policies
curl -H 'x-api-key: your-secret' http://127.0.0.1:8000/metrics
curl -H 'x-api-key: your-secret' http://127.0.0.1:8000/calibration
```

Analyze request:

```bash
curl -X POST http://127.0.0.1:8000/analyze \
  -H 'Content-Type: application/json' \
  -H 'x-api-key: your-secret' \
  -d '{"text":"I am the CEO, wire funds now.","modality":"text","identity_claim":"CEO Name","profile":"industry_low_fp"}'
```

## Benchmark and readiness scorecard

Use a labeled CSV (`label` required, plus `input_path` or `text`).
Template: `examples/benchmark_template.csv`

```bash
python3 -m aip.cli evaluate \
  --dataset examples/benchmark_template.csv \
  --profile industry_low_fp \
  --threshold 0.65 \
  --target-fpr 0.03 \
  --output evaluation_report.json \
  --scorecard-json scorecard.json \
  --scorecard-md scorecard.md \
  --calibration-output calibration.json \
  --pretty
```

Outputs:
- `evaluation_report.json` with FPR/TPR/precision/recall and recommended threshold
- `scorecard.json` and `scorecard.md` with readiness grade and blocking gaps
- `calibration.json` with per-profile and per-modality thresholds for runtime decisions

## Provenance sidecar format

If present, `<asset>.prov.json` is used:

```json
{
  "signature_valid": true,
  "issuer": "trusted-camera-vendor",
  "generated_by": "in-camera-signing",
  "chain_of_custody": ["capture", "upload", "archive"]
}
```

Optional C2PA marker sidecar: `<asset>.c2pa`

## Production docs

- `DEPLOYMENT.md`
- `SECURITY.md`
- `MARKET_READINESS.md`

## Testing

```bash
python3 -m unittest discover -s tests -p 'test_*.py' -v
```

Pre-release (privacy + quality gate):

```bash
bash scripts/pre_release_check.sh
```

## Notes

- This is a robust triage/orchestration platform, not a standalone legal verdict engine.
- Production trust requires periodic threshold calibration on representative labeled datasets.
- Missing optional tools (for example `ffprobe`, `ffmpeg`) degrade observability but do not hard-fail requests.
