# Contributing

Thanks for contributing.

## Quick start

1. Fork and clone the repository.
2. Install development dependencies:

```bash
python3 -m pip install -e '.[forensics,api,test]'
```

3. Run checks before opening a PR:

```bash
python3 -m unittest discover -s tests -p 'test_*.py' -v
bash scripts/pre_release_check.sh
```

## Pull request guidelines

- Keep PRs focused and small.
- Add or update tests for behavior changes.
- Update docs (`README.md`, deployment/security docs) when behavior changes.
- Avoid committing secrets, local env files, generated reports, and logs.

## Commit style (recommended)

Use clear commit titles, for example:
- `feat: add calibration endpoint`
- `fix: tighten file path guard`
- `docs: update deployment checklist`
- `test: add rate limiter coverage`

## Reporting issues

Please include:
- environment details (OS, Python version)
- exact command/request used
- expected behavior vs actual behavior
- sanitized logs or stack traces
