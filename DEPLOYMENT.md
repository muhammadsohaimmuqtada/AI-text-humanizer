# Deployment Guide

## 1. Install

```bash
python3 -m pip install -e '.[api]'
python3 -m nltk.downloader wordnet punkt_tab averaged_perceptron_tagger_eng
```

## 2. Configure

Set environment variables (from `.env.example`):
- `AIP_API_KEY`
- `AIP_MAX_TEXT_CHARS`
- `AIP_RATE_LIMIT_PER_MIN`
- `AIP_AUDIT_LOG_PATH`
- `AIP_CALIBRATION_FILE` (optional)

## 3. Preflight checks

```bash
aip preflight --pretty
aip doctor --pretty
```

## 4. Run API

```bash
aip serve --host 0.0.0.0 --port 8000 --workers 2 --log-level info
```

## 5. Probe endpoints

```bash
curl http://127.0.0.1:8000/healthz
curl http://127.0.0.1:8000/readyz
curl -H 'x-api-key: YOUR_API_KEY' http://127.0.0.1:8000/policies
curl -H 'x-api-key: YOUR_API_KEY' http://127.0.0.1:8000/metrics
```

## 6. Run the GUI (alternative)

```bash
aip-gui
```
