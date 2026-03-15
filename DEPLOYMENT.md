# Deployment Guide

## 1. Install

```bash
python3 -m pip install -e '.[forensics,api]'
```

## 2. Configure

Set environment variables (from `.env.example`):
- `AIP_API_KEY`
- `AIP_ALLOWED_INPUT_ROOTS`
- `AIP_MAX_FILE_SIZE_MB`
- `AIP_RATE_LIMIT_PER_MIN`
- `AIP_AUDIT_LOG_PATH`
- `AIP_CALIBRATION_FILE` (optional but recommended after benchmark tuning)

## 3. Preflight checks

```bash
python3 -m aip.cli preflight --pretty
python3 -m aip.cli doctor --pretty
```

## 4. Run API

```bash
python3 -m aip.cli serve --host 0.0.0.0 --port 8000 --workers 2 --log-level info
```

## 5. Probe endpoints

```bash
curl http://127.0.0.1:8000/healthz
curl http://127.0.0.1:8000/readyz
curl -H 'x-api-key: your-secret' http://127.0.0.1:8000/policies
curl -H 'x-api-key: your-secret' http://127.0.0.1:8000/metrics
curl -H 'x-api-key: your-secret' http://127.0.0.1:8000/calibration
```

## 6. Evaluate with benchmark data before launch

```bash
python3 -m aip.cli evaluate \
  --dataset /path/to/labeled.csv \
  --profile industry_low_fp \
  --threshold 0.65 \
  --target-fpr 0.03 \
  --output evaluation_report.json \
  --scorecard-json scorecard.json \
  --scorecard-md scorecard.md \
  --calibration-output calibration.json \
  --pretty
```

Launch criteria recommendation:
- `scorecard` gate `fpr<=0.03` must pass
- `sample_count>=200` should pass per critical modality
- `inconclusive_rate<=0.25` should pass
