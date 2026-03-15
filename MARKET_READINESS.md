# Market Readiness Framework

AIP should be treated as market-ready only when both are true:

1. Production hardening checks pass (`aip preflight`).
2. Benchmark scorecard gates pass on representative data.
3. Latest benchmark-derived `calibration.json` is deployed (`AIP_CALIBRATION_FILE`).

## Minimum go-live criteria

- `precision>=0.90`
- `fpr<=0.03`
- `sample_count>=200` (per critical modality recommended)
- `avg_coverage>=0.70`
- `avg_quality>=0.55`
- `inconclusive_rate<=0.25`

## Launch phases

- Alpha: internal validation, limited claims
- Pilot Ready: controlled customers with monitored decisions
- Market Candidate: broad launch with stable thresholds, weekly recalibration cadence

## Claims discipline

Do not claim:
- "perfect detection"
- "zero false positives"
- "court-proof certainty"

Safe claim style:
- "risk-based authenticity scoring with provenance and human-review workflows"
