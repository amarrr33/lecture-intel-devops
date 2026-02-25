from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional, List
import os

AUDIO_EXTS = {".wav", ".mp3", ".webm", ".m4a", ".aac", ".flac", ".ogg"}


def _run(cmd: List[str]) -> None:
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"Command failed:\n{p.stderr}")


def _ensure_tool(name: str):
    if not shutil.which(name):
        raise RuntimeError(f"{name} not installed")


# ---------------------------
# NORMALIZE AUDIO
# ---------------------------
def normalize_to_wav16k(in_path: str | Path, out_path: str | Path) -> Path:
    _ensure_tool("ffmpeg")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg", "-y",
        "-i", str(in_path),
        "-ac", "1",
        "-ar", "16000",
        str(out_path)
    ]
    _run(cmd)

    return out_path


# ---------------------------
# 🔥 WHISPER FIXED
# ---------------------------
def load_whisper_model(model_size: str):
    import whisper

    try:
        return whisper.load_model(model_size)
    except RuntimeError:
        # 🔥 fix corrupted download
        cache = Path.home() / ".cache" / "whisper"
        if cache.exists():
            shutil.rmtree(cache)

        return whisper.load_model(model_size)


def transcribe_whisper(
    audio_path: str | Path,
    model_size: str = "base",
    out_dir: Optional[str | Path] = None,
) -> Dict[str, Any]:

    audio_path = Path(audio_path)

    if out_dir is None:
        out_dir = audio_path.parent / "outputs"

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    wav_path = out_dir / "audio_16k.wav"

    # normalize always (fixes random bugs)
    normalize_to_wav16k(audio_path, wav_path)

    # load model safely
    model = load_whisper_model(model_size)

    result = model.transcribe(str(wav_path), fp16=False)

    segments = []
    for s in result.get("segments", []):
        text = (s.get("text") or "").strip()
        if text:
            segments.append({
                "start": float(s["start"]),
                "end": float(s["end"]),
                "text": text
            })

    return {
        "language": result.get("language"),
        "text": (result.get("text") or "").strip(),
        "segments": segments,
        "audio_wav16k": str(wav_path),
    }