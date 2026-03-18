# Contributing

Thanks for your interest in contributing!

## Quick start

1. Fork and clone the repository.
2. Create a virtual environment and install dev dependencies:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -e '.[api,test]'
python -m nltk.downloader wordnet punkt_tab averaged_perceptron_tagger_eng
```

3. Run tests before opening a PR:

```bash
python -m pytest tests/ -v
```

## Pull request guidelines

- Keep PRs focused and small.
- Add or update tests for behavior changes.
- Update docs (`README.md`, deployment/security docs) when behavior changes.
- Avoid committing secrets, local env files, generated reports, and logs.

## Commit style (recommended)

Use clear commit titles, for example:
- `feat: add calibration endpoint`
- `fix: tighten synonym blacklist`
- `docs: update deployment checklist`
- `test: add contraction insertion coverage`

## Reporting issues

Please include:
- Environment details (OS, Python version)
- Exact command/request used
- Expected behavior vs actual behavior
- Sanitized logs or stack traces
