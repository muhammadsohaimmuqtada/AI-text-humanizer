from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def load_calibration(path: Optional[str]) -> Optional[Dict[str, Any]]:
    if not path:
        return None
    p = Path(path).expanduser().resolve()
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def save_calibration(payload: Dict[str, Any], path: str) -> str:
    p = Path(path).expanduser().resolve()
    p.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return str(p)


def resolve_threshold_override(
    calibration: Optional[Dict[str, Any]],
    profile: str,
    modality: str,
) -> Optional[float]:
    if not calibration:
        return None

    try:
        profiles = calibration.get("profiles", {})
        prof = profiles.get(profile, {}) if isinstance(profiles, dict) else {}
        if isinstance(prof, dict):
            modalities = prof.get("modalities", {})
            if isinstance(modalities, dict):
                mod = modalities.get(modality, {})
                if isinstance(mod, dict) and "threshold" in mod:
                    return _clamp(float(mod["threshold"]))
            if "default_threshold" in prof:
                return _clamp(float(prof["default_threshold"]))
        if "default_threshold" in calibration:
            return _clamp(float(calibration["default_threshold"]))
    except Exception:
        return None

    return None
