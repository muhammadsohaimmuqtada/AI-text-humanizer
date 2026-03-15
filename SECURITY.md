# Security Practices

## Built-in controls

- API key authentication via `AIP_API_KEY`
- Token-bucket rate limiting
- Input path allow-list restriction via `AIP_ALLOWED_INPUT_ROOTS`
- Max text and file size constraints
- Request IDs and structured audit logs (`AIP_AUDIT_LOG_PATH`)

## Recommended deployment posture

- Always set `AIP_API_KEY` in non-private environments
- Restrict `AIP_ALLOWED_INPUT_ROOTS` to minimal required directories
- Rotate API keys periodically
- Protect audit logs and metrics endpoints at network boundary
- Run benchmark evaluation before promoting thresholds

## Incident response basics

- Keep audit logs for investigation and threshold tuning
- On abuse spike:
  - lower rate limits
  - tighten allowed paths
  - force `industry_low_fp` profile
- Re-run scorecard after any threshold or policy change

## Limitations

- This system is decision-support and triage infrastructure.
- High-stakes enforcement should include analyst review and provenance policy.
