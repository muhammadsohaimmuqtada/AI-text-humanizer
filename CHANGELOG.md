# Changelog

## 1.0.0 - 2026-03-18

### Added
- **8-stage humanization pipeline** (expanded from 4 stages)
- Contraction insertion (50+ patterns: "do not" → "don't", etc.)
- Clause reordering (moves prepositional phrases to vary structure)
- Sentence splitting (breaks long compound sentences at conjunctions)
- Discourse filler insertion (25 natural human phrases)
- Expanded AI marker list (~90 patterns including modern GPT-era markers)
- Enhanced burstiness variation (16 conjunction styles, up from 5)
- Improved synonym substitution (no hard rate cap, expanded blacklist)
- NLTK contraction spacing post-processing fix
- Comprehensive test suite (78 tests + 15 subtests)

### Changed
- GUI bypass presets updated with rates for all 8 transforms
- Synonym blacklist expanded to ~100 words for robustness
- README rewritten with full pipeline documentation

## 0.3.0 - 2026-03-15

### Added
- Hardened API controls (auth, rate limits, audit logging, path and size guardrails)
- Readiness endpoints and metrics
- Benchmark evaluation and scorecard workflow
- Calibration bundle generation and runtime usage
- CI workflow for automated checks
- Deployment/security docs

### Changed
- Secure-by-default input root handling (`/tmp` unless explicitly configured)

## 0.2.0 - 2026-03-15

### Added
- API service mode
- Evaluation pipeline and readiness scoring
- Dark-themed Tkinter GUI with bypass strength presets

## 0.1.0 - 2026-03-15

### Added
- Initial text humanization CLI with marker stripping and synonym substitution
