from __future__ import annotations

import hashlib
import io
import json
import math
import mimetypes
import re
import shutil
import subprocess
import tempfile
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .types import DetectionResult, Signal

try:
    import numpy as np
except Exception:  # pragma: no cover - optional dependency
    np = None

try:
    from PIL import Image
except Exception:  # pragma: no cover - optional dependency
    Image = None

try:
    from scipy.io import wavfile
except Exception:  # pragma: no cover - optional dependency
    wavfile = None


_TEXT_AI_MARKERS = (
    "as an ai language model",
    "i cannot browse",
    "i do not have real-time",
    "i'm unable to",
    "i can not provide legal advice",
)

_SYNTH_TOOL_KEYWORDS = (
    "stable diffusion",
    "midjourney",
    "dall-e",
    "firefly",
    "synthesia",
    "runway",
    "deepfacelab",
    "faceswap",
    "elevenlabs",
    "play.ht",
    "murf",
    "descript",
)

_EDITING_TOOL_KEYWORDS = (
    "photoshop",
    "gimp",
    "lightroom",
    "after effects",
    "premiere",
    "final cut",
    "davinci",
    "capcut",
    "ffmpeg",
)

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".tiff", ".bmp"}
_AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg"}
_VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}


def runtime_capabilities() -> Dict[str, bool]:
    return {
        "exiftool": shutil.which("exiftool") is not None,
        "ffprobe": shutil.which("ffprobe") is not None,
        "ffmpeg": shutil.which("ffmpeg") is not None,
        "numpy": np is not None,
        "pillow": Image is not None,
        "scipy_wav": wavfile is not None,
    }


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def guess_modality(path: Optional[Path], explicit: str) -> str:
    if explicit and explicit != "auto":
        return explicit
    if not path:
        return "text"

    ext = path.suffix.lower()
    if ext in _IMAGE_EXTS:
        return "image"
    if ext in _AUDIO_EXTS:
        return "audio"
    if ext in _VIDEO_EXTS:
        return "video"

    mime, _ = mimetypes.guess_type(str(path))
    if mime:
        if mime.startswith("image/"):
            return "image"
        if mime.startswith("audio/"):
            return "audio"
        if mime.startswith("video/"):
            return "video"
        if mime.startswith("text/"):
            return "text"
    return "binary"


def _clamp(x: float) -> float:
    return max(0.0, min(1.0, x))


def _safe_json_command(cmd: List[str], timeout: int = 10) -> Tuple[Optional[object], Optional[str]]:
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=timeout)
    except FileNotFoundError:
        return None, f"{cmd[0]} unavailable"
    except subprocess.SubprocessError:
        return None, f"{cmd[0]} failed"

    try:
        parsed = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return None, f"{cmd[0]} output parse error"
    return parsed, None


def _byte_entropy(sample: bytes) -> float:
    if not sample:
        return 0.0
    counts = Counter(sample)
    n = len(sample)
    entropy = -sum((c / n) * math.log2(c / n) for c in counts.values())
    return entropy / 8.0


def _header_matches_extension(path: Path, data: bytes) -> bool:
    ext = path.suffix.lower()
    if ext in {".jpg", ".jpeg"}:
        return data.startswith(b"\xff\xd8\xff")
    if ext == ".png":
        return data.startswith(b"\x89PNG\r\n\x1a\n")
    if ext in {".mp4", ".m4v", ".mov"}:
        return b"ftyp" in data[:64]
    if ext == ".wav":
        return data.startswith(b"RIFF") and b"WAVE" in data[:16]
    return True


def _flatten_metadata_text(meta: Dict[str, object]) -> str:
    parts: List[str] = []
    for value in meta.values():
        if isinstance(value, (str, int, float, bool)):
            parts.append(str(value).lower())
    return " | ".join(parts)


def _metadata_signals(path: Path, modality: str) -> Tuple[List[Signal], float, float, float, float]:
    signals: List[Signal] = []
    caps = runtime_capabilities()

    if not caps["exiftool"]:
        signals.append(Signal("metadata_probe", 0.2, "exiftool unavailable"))
        return signals, 0.0, 0.0, 0.05, 0.2

    payload, err = _safe_json_command(["exiftool", "-j", str(path)], timeout=8)
    if err:
        signals.append(Signal("metadata_probe", 0.3, err))
        return signals, 0.0, 0.0, 0.08, 0.3

    meta_list = payload if isinstance(payload, list) else []
    meta = meta_list[0] if meta_list and isinstance(meta_list[0], dict) else {}
    meta_text = _flatten_metadata_text(meta)

    synth_hits = sum(1 for kw in _SYNTH_TOOL_KEYWORDS if kw in meta_text)
    edit_hits = sum(1 for kw in _EDITING_TOOL_KEYWORDS if kw in meta_text)

    # Keep missing metadata logic modality-aware to avoid over-penalizing
    # assets that naturally do not carry camera fields.
    key_groups = {
        "image": ("CreateDate", "DateTimeOriginal", "ModifyDate"),
        "audio": ("Duration", "AudioSampleRate", "SampleRate"),
        "video": ("Duration", "VideoFrameRate", "CompressorName", "Encoder"),
        "binary": ("FileType",),
    }
    expected = key_groups.get(modality, key_groups["binary"])
    present = sum(1 for key in expected if key in meta)
    missing_score = _clamp((len(expected) - present) / max(1, len(expected)))

    synth_score = _clamp(synth_hits / 2)
    edit_score = _clamp(edit_hits / 2)

    signals.append(Signal("metadata_synth_tool_hits", synth_score, f"synth_hits={synth_hits}"))
    signals.append(Signal("metadata_edit_tool_hits", edit_score, f"edit_hits={edit_hits}"))
    signals.append(
        Signal(
            "metadata_missing_expected",
            missing_score,
            f"present_expected={present}/{len(expected)}",
        )
    )

    return signals, 0.35 * edit_score, 0.45 * synth_score, 0.1 * missing_score, 0.8


def _ffprobe_summary(path: Path) -> Tuple[Optional[dict], Optional[str]]:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        str(path),
    ]
    payload, err = _safe_json_command(cmd, timeout=8)
    if err:
        return None, err
    return payload if isinstance(payload, dict) else None, None


def _image_forensics(path: Path) -> Tuple[List[Signal], float, float, float, float, float]:
    signals: List[Signal] = []
    if np is None or Image is None:
        signals.append(Signal("image_forensics", 0.2, "numpy/pillow unavailable"))
        return signals, 0.0, 0.0, 0.1, 0.2, 0.35

    try:
        img = Image.open(path).convert("RGB")
    except Exception:
        signals.append(Signal("image_forensics", 0.3, "failed to decode image"))
        return signals, 0.1, 0.0, 0.3, 0.4, 0.25

    arr = np.asarray(img, dtype=np.float32)
    h, w, _ = arr.shape
    gray = np.asarray(img.convert("L"), dtype=np.float32)

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    buf.seek(0)
    recompressed = np.asarray(Image.open(buf).convert("RGB"), dtype=np.float32)
    ela = np.abs(arr - recompressed)
    ela_mean = float(np.mean(ela) / 255.0)

    block = 32
    patch_std = []
    for y in range(0, max(block, h - block + 1), block):
        for x in range(0, max(block, w - block + 1), block):
            patch = gray[y : min(y + block, h), x : min(x + block, w)]
            if patch.size > 0:
                patch_std.append(float(np.std(patch)))

    if patch_std:
        patch_std_arr = np.asarray(patch_std, dtype=np.float32)
        patch_inconsistency = float(np.std(patch_std_arr) / (np.mean(patch_std_arr) + 1e-6))
    else:
        patch_inconsistency = 0.0

    gx = np.diff(gray, axis=1)
    gy = np.diff(gray, axis=0)
    edge_mag = np.sqrt(gx[:-1, :] ** 2 + gy[:, :-1] ** 2)
    edge_density = float(np.mean(edge_mag > 18.0)) if edge_mag.size else 0.0

    sat = (np.max(arr, axis=2) - np.min(arr, axis=2)) / (np.max(arr, axis=2) + 1.0)
    saturation_mean = float(np.mean(sat))

    small_image = 1.0 if min(h, w) < 96 else 0.0

    signals.append(Signal("ela_mean", _clamp(ela_mean * 4), f"ela_mean={ela_mean:.4f}"))
    signals.append(
        Signal("patch_noise_inconsistency", _clamp(patch_inconsistency / 2), f"patch_inconsistency={patch_inconsistency:.3f}")
    )
    signals.append(Signal("edge_density", _clamp(edge_density * 2), f"edge_density={edge_density:.3f}"))
    signals.append(Signal("saturation_mean", _clamp(saturation_mean), f"saturation_mean={saturation_mean:.3f}"))
    resolution_quality = _clamp((h * w) / float(720 * 720))
    contrast_quality = _clamp(float(np.std(gray)) / 70.0)
    image_quality = _clamp(0.6 * resolution_quality + 0.4 * contrast_quality)
    signals.append(
        Signal(
            "quality_image_observability",
            image_quality,
            f"resolution={w}x{h}, contrast_std={float(np.std(gray)):.2f}",
        )
    )

    manipulation = _clamp(0.35 * _clamp(ela_mean * 3.5) + 0.35 * _clamp(patch_inconsistency / 2) + 0.2 * small_image)
    synthetic = _clamp(
        0.3 * _clamp(ela_mean * 2.5)
        + 0.25 * _clamp((0.08 - edge_density) / 0.08)
        + 0.2 * _clamp((0.85 - saturation_mean) / 0.85)
    )
    anomaly = _clamp(0.25 * small_image + 0.2 * _clamp(patch_inconsistency / 2.5))

    return signals, manipulation, synthetic, anomaly, 0.9, image_quality


def _audio_features_wav(path: Path) -> Tuple[List[Signal], float, float, float, float, float]:
    signals: List[Signal] = []
    if np is None or wavfile is None:
        signals.append(Signal("audio_forensics", 0.25, "numpy/scipy wav unavailable"))
        return signals, 0.0, 0.0, 0.1, 0.25, 0.3

    if path.suffix.lower() != ".wav":
        signals.append(Signal("audio_codec_support", 0.2, "advanced analysis supports wav directly"))
        return signals, 0.05, 0.05, 0.1, 0.35, 0.45

    try:
        sr, x = wavfile.read(path)
    except Exception:
        signals.append(Signal("audio_decode", 0.35, "failed to decode wav"))
        return signals, 0.1, 0.1, 0.25, 0.4, 0.25

    if x.ndim > 1:
        x = np.mean(x, axis=1)

    if np.issubdtype(x.dtype, np.integer):
        max_val = max(abs(np.iinfo(x.dtype).min), abs(np.iinfo(x.dtype).max))
        x = x.astype(np.float32) / float(max_val)
    else:
        x = x.astype(np.float32)

    x = np.clip(x, -1.0, 1.0)
    n = x.shape[0]
    duration = n / max(sr, 1)

    clipping_ratio = float(np.mean(np.abs(x) > 0.985)) if n else 0.0
    silence_ratio = float(np.mean(np.abs(x) < 0.003)) if n else 1.0

    if n > 1:
        zcr = float(np.mean(np.signbit(x[:-1]) != np.signbit(x[1:])))
    else:
        zcr = 0.0

    nfft = 1024
    hop = 512
    window = np.hanning(nfft).astype(np.float32)
    flatness_values: List[float] = []
    rms_values: List[float] = []

    for i in range(0, max(1, n - nfft), hop):
        frame = x[i : i + nfft]
        if frame.shape[0] < nfft:
            break
        frame_w = frame * window
        spec = np.abs(np.fft.rfft(frame_w)) + 1e-9
        flatness = float(np.exp(np.mean(np.log(spec))) / np.mean(spec))
        flatness_values.append(flatness)
        rms_values.append(float(np.sqrt(np.mean(frame_w**2))))

    spectral_flatness = float(np.median(flatness_values)) if flatness_values else 0.0
    energy_var = float(np.std(rms_values)) if rms_values else 0.0

    signals.append(Signal("audio_duration_s", _clamp(duration / 60), f"duration={duration:.2f}s"))
    signals.append(Signal("audio_clipping_ratio", _clamp(clipping_ratio * 12), f"clipping_ratio={clipping_ratio:.4f}"))
    signals.append(Signal("audio_silence_ratio", _clamp(silence_ratio), f"silence_ratio={silence_ratio:.3f}"))
    signals.append(Signal("audio_zcr", _clamp(zcr * 8), f"zcr={zcr:.4f}"))
    signals.append(Signal("audio_spectral_flatness", _clamp(spectral_flatness * 2.5), f"flatness={spectral_flatness:.4f}"))
    duration_quality = _clamp(duration / 8.0)
    dynamic_quality = _clamp(min(1.0, (energy_var + 0.01) / 0.08))
    audio_quality = _clamp(0.55 * duration_quality + 0.45 * dynamic_quality)
    signals.append(
        Signal(
            "quality_audio_observability",
            audio_quality,
            f"duration={duration:.2f}s, energy_var={energy_var:.4f}",
        )
    )

    manipulation = _clamp(0.35 * _clamp(clipping_ratio * 10) + 0.2 * _clamp((0.2 - duration) / 0.2))
    synthetic = _clamp(
        0.3 * _clamp((0.06 - energy_var) / 0.06)
        + 0.3 * _clamp((0.055 - zcr) / 0.055)
        + 0.2 * _clamp((spectral_flatness - 0.35) / 0.4)
    )
    anomaly = _clamp(0.2 * _clamp((0.1 - duration) / 0.1) + 0.2 * _clamp((silence_ratio - 0.9) / 0.1))

    return signals, manipulation, synthetic, anomaly, 0.9, audio_quality


def _average_hash(pil_image: "Image.Image") -> int:
    g = pil_image.convert("L").resize((8, 8))
    arr = np.asarray(g, dtype=np.float32)
    bits = arr > np.mean(arr)
    out = 0
    for bit in bits.flatten():
        out = (out << 1) | int(bool(bit))
    return out


def _hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def _video_forensics(path: Path) -> Tuple[List[Signal], float, float, float, float, float]:
    signals: List[Signal] = []
    caps = runtime_capabilities()

    ffprobe_data, ffprobe_err = _ffprobe_summary(path)
    if ffprobe_err:
        signals.append(Signal("video_probe", 0.2, ffprobe_err))

    video_streams = []
    audio_streams = []
    fps_val = 0.0
    short_duration = 0.0

    if ffprobe_data:
        streams = ffprobe_data.get("streams", []) if isinstance(ffprobe_data.get("streams"), list) else []
        video_streams = [s for s in streams if s.get("codec_type") == "video"]
        audio_streams = [s for s in streams if s.get("codec_type") == "audio"]

        if video_streams:
            fr = str(video_streams[0].get("avg_frame_rate", "0/1"))
            if "/" in fr:
                num, den = fr.split("/", 1)
                try:
                    fps_val = float(num) / max(float(den), 1.0)
                except ValueError:
                    fps_val = 0.0

        try:
            duration = float(ffprobe_data.get("format", {}).get("duration", "0") or "0")
        except ValueError:
            duration = 0.0
        short_duration = 1.0 if duration < 0.3 else 0.0

        signals.append(Signal("video_stream_count", _clamp(len(video_streams) / 3), f"video_streams={len(video_streams)}"))
        signals.append(Signal("audio_stream_count", _clamp(len(audio_streams) / 3), f"audio_streams={len(audio_streams)}"))
        signals.append(Signal("video_fps", _clamp(fps_val / 120), f"fps={fps_val:.2f}"))

    frame_motion_signal = 0.0
    extracted_frames = 0
    if caps["ffmpeg"] and Image is not None and np is not None:
        try:
            with tempfile.TemporaryDirectory(prefix="aip_frames_") as td:
                frame_pattern = str(Path(td) / "frame_%03d.jpg")
                cmd = [
                    "ffmpeg",
                    "-v",
                    "error",
                    "-i",
                    str(path),
                    "-vf",
                    "fps=2",
                    "-frames:v",
                    "8",
                    frame_pattern,
                ]
                subprocess.run(cmd, capture_output=True, text=True, timeout=15, check=True)

                frame_paths = sorted(Path(td).glob("frame_*.jpg"))
                extracted_frames = len(frame_paths)
                hashes = []
                for fp in frame_paths:
                    hashes.append(_average_hash(Image.open(fp)))

                if len(hashes) >= 2:
                    distances = [_hamming(hashes[i], hashes[i + 1]) for i in range(len(hashes) - 1)]
                    avg_dist = sum(distances) / len(distances)
                    frame_motion_signal = _clamp((6.0 - avg_dist) / 6.0)
                    signals.append(Signal("video_frame_hash_stability", frame_motion_signal, f"avg_hamming={avg_dist:.2f}"))
        except Exception:
            signals.append(Signal("video_frame_analysis", 0.2, "ffmpeg frame extraction failed"))
    else:
        signals.append(Signal("video_frame_analysis", 0.15, "ffmpeg/numpy/pillow unavailable"))

    manipulation = _clamp(0.3 * short_duration + 0.3 * frame_motion_signal + 0.2 * (1.0 if not audio_streams else 0.0))
    synthetic = _clamp(0.25 * frame_motion_signal + 0.2 * (1.0 if fps_val > 60.0 else 0.0))
    anomaly = _clamp(0.3 * short_duration + 0.2 * (1.0 if not video_streams else 0.0))

    coverage = 0.6
    if ffprobe_data:
        coverage += 0.2
    if caps["ffmpeg"] and Image is not None and np is not None:
        coverage += 0.2

    frame_quality = _clamp(extracted_frames / 8.0)
    stream_quality = 0.6 if ffprobe_data else 0.3
    video_quality = _clamp(0.55 * frame_quality + 0.45 * stream_quality)
    signals.append(
        Signal(
            "quality_video_observability",
            video_quality,
            f"frames_sampled={extracted_frames}, ffprobe_present={ffprobe_data is not None}",
        )
    )
    return signals, manipulation, synthetic, anomaly, _clamp(coverage), video_quality


def analyze_text(text: str, identity_claim: Optional[str]) -> DetectionResult:
    cleaned = text.strip().lower()
    words = re.findall(r"[a-zA-Z0-9']+", cleaned)
    sentences = [s.strip() for s in re.split(r"[.!?]+", cleaned) if s.strip()]

    token_count = len(words)
    unique_count = len(set(words)) if words else 1
    ttr = unique_count / max(1, token_count)

    sentence_lengths = [len(re.findall(r"[a-zA-Z0-9']+", s)) for s in sentences]
    mean_sent = sum(sentence_lengths) / max(1, len(sentence_lengths))
    var_sent = 0.0
    if sentence_lengths:
        var_sent = sum((x - mean_sent) ** 2 for x in sentence_lengths) / len(sentence_lengths)

    burstiness = var_sent / max(1.0, mean_sent)

    bigrams = [tuple(words[i : i + 2]) for i in range(max(0, token_count - 1))]
    repeated_bigrams = sum(1 for _, c in Counter(bigrams).items() if c > 1)
    repetition_ratio = repeated_bigrams / max(1, len(bigrams))

    ai_marker_hits = sum(1 for marker in _TEXT_AI_MARKERS if marker in cleaned)
    punctuation_density = len(re.findall(r"[,:;()\[\]-]", text)) / max(1, len(text))
    zero_width_hits = len(re.findall(r"[\u200B-\u200D\uFEFF]", text))

    synthetic = _clamp(
        0.3 * (1 - _clamp(ttr))
        + 0.25 * _clamp(repetition_ratio * 4)
        + 0.2 * _clamp((0.2 - burstiness) / 0.2)
        + 0.15 * min(1.0, ai_marker_hits / 2)
    )

    manipulation = _clamp(
        0.25 * _clamp(repetition_ratio * 4)
        + 0.2 * (1.0 if punctuation_density > 0.12 else 0.0)
        + 0.25 * _clamp(zero_width_hits / 2)
    )

    impersonation = 0.05
    if identity_claim:
        lc_claim = identity_claim.lower()
        claim_mentions = 1.0 if lc_claim in cleaned else 0.0
        first_person = 1.0 if re.search(r"\bi\b|\bmy\b|\bme\b", cleaned) else 0.0
        impersonation = _clamp(0.1 + 0.45 * claim_mentions * first_person + 0.15 * synthetic)

    anomaly = _clamp(0.2 * _clamp(repetition_ratio * 4) + 0.25 * _clamp(zero_width_hits / 2) + 0.15 * (1.0 if token_count < 3 else 0.0))

    signals = [
        Signal("token_count", _clamp(token_count / 800), f"token_count={token_count}"),
        Signal("type_token_ratio", 1 - _clamp(ttr), f"ttr={ttr:.3f}"),
        Signal("repetition_ratio", _clamp(repetition_ratio * 4), f"repetition_ratio={repetition_ratio:.3f}"),
        Signal("sentence_burstiness", _clamp((0.8 - burstiness) / 0.8), f"burstiness={burstiness:.3f}"),
        Signal("ai_marker_hits", _clamp(ai_marker_hits / 3), f"ai_marker_hits={ai_marker_hits}"),
        Signal("zero_width_chars", _clamp(zero_width_hits / 2), f"zero_width_hits={zero_width_hits}"),
    ]
    length_quality = _clamp(token_count / 80.0)
    structure_quality = _clamp(len(sentences) / 8.0)
    lexical_quality = _clamp(ttr / 0.5)
    text_quality = _clamp(0.5 * length_quality + 0.3 * structure_quality + 0.2 * lexical_quality)
    signals.append(
        Signal(
            "quality_text_observability",
            text_quality,
            f"tokens={token_count}, sentences={len(sentences)}, ttr={ttr:.3f}",
        )
    )

    return DetectionResult(
        manipulation_likelihood=manipulation,
        synthetic_likelihood=synthetic,
        impersonation_likelihood=impersonation,
        anomaly_likelihood=anomaly,
        coverage=0.95,
        quality=text_quality,
        signals=signals,
    )


def analyze_file(path: Path, modality: str, identity_claim: Optional[str]) -> DetectionResult:
    data = path.read_bytes()
    sample = data[: min(len(data), 2 * 1024 * 1024)]

    entropy_norm = _byte_entropy(sample)
    header_ok = _header_matches_extension(path, sample)
    size_mb = len(data) / (1024 * 1024)

    manipulation = _clamp(0.25 * (0 if header_ok else 1) + 0.15 * (1 if entropy_norm < 0.2 or entropy_norm > 0.98 else 0))
    synthetic = _clamp(0.2 * (1 if entropy_norm > 0.94 else 0) + 0.1 * (1 if size_mb < 0.02 else 0))
    anomaly = _clamp(0.2 * (0 if header_ok else 1) + 0.1 * (1 if entropy_norm < 0.1 else 0))

    signals: List[Signal] = [
        Signal("header_consistency", 0 if header_ok else 1, f"header_matches_extension={header_ok}"),
        Signal("byte_entropy", entropy_norm, f"entropy_norm={entropy_norm:.3f}"),
        Signal("asset_size_mb", _clamp(size_mb / 200), f"size_mb={size_mb:.3f}"),
    ]

    coverage_parts = [0.35]
    base_quality = _clamp(0.35 + min(0.5, size_mb / 8.0))
    quality_parts = [base_quality]

    meta_signals, meta_m, meta_s, meta_a, meta_cov = _metadata_signals(path, modality)
    signals.extend(meta_signals)
    manipulation = _clamp(manipulation + meta_m)
    synthetic = _clamp(synthetic + meta_s)
    anomaly = _clamp(anomaly + meta_a)
    coverage_parts.append(meta_cov)

    if modality == "image":
        img_signals, img_m, img_s, img_a, img_cov, img_q = _image_forensics(path)
        signals.extend(img_signals)
        manipulation = _clamp(0.65 * manipulation + 0.35 * img_m)
        synthetic = _clamp(0.55 * synthetic + 0.45 * img_s)
        anomaly = _clamp(0.6 * anomaly + 0.4 * img_a)
        coverage_parts.append(img_cov)
        quality_parts.append(img_q)
    elif modality == "audio":
        aud_signals, aud_m, aud_s, aud_a, aud_cov, aud_q = _audio_features_wav(path)
        signals.extend(aud_signals)
        manipulation = _clamp(0.65 * manipulation + 0.35 * aud_m)
        synthetic = _clamp(0.55 * synthetic + 0.45 * aud_s)
        anomaly = _clamp(0.6 * anomaly + 0.4 * aud_a)
        coverage_parts.append(aud_cov)
        quality_parts.append(aud_q)
    elif modality == "video":
        vid_signals, vid_m, vid_s, vid_a, vid_cov, vid_q = _video_forensics(path)
        signals.extend(vid_signals)
        manipulation = _clamp(0.6 * manipulation + 0.4 * vid_m)
        synthetic = _clamp(0.55 * synthetic + 0.45 * vid_s)
        anomaly = _clamp(0.55 * anomaly + 0.45 * vid_a)
        coverage_parts.append(vid_cov)
        quality_parts.append(vid_q)

    impersonation = 0.06
    if identity_claim:
        impersonation = _clamp(0.18 + 0.35 * synthetic + 0.25 * manipulation)

    coverage = _clamp(sum(coverage_parts) / len(coverage_parts))
    quality = _clamp(sum(quality_parts) / len(quality_parts))

    return DetectionResult(
        manipulation_likelihood=manipulation,
        synthetic_likelihood=synthetic,
        impersonation_likelihood=impersonation,
        anomaly_likelihood=anomaly,
        coverage=coverage,
        quality=quality,
        signals=signals,
    )
