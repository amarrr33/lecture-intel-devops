from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Optional, Dict, Any, List


AUDIO_EXTS = {".wav", ".mp3", ".webm", ".m4a", ".aac", ".flac", ".ogg"}


def _run(cmd: List[str]) -> None:
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(
            f"Command failed:\n{cmd}\n\nSTDOUT:\n{p.stdout}\n\nSTDERR:\n{p.stderr}"
        )


def _ensure_tool(name: str) -> str:
    path = shutil.which(name)
    if not path:
        raise RuntimeError(
            f"'{name}' not found in PATH. Install it and reopen terminal/VS Code."
        )
    return path


def normalize_to_wav16k(in_path: str | Path, out_path: str | Path) -> Path:
    """
    Convert ANY audio to Whisper-friendly WAV: 16kHz, mono.
    Requires: ffmpeg in PATH.
    """
    _ensure_tool("ffmpeg")
    in_path = str(in_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = ["ffmpeg", "-y", "-i", in_path, "-ac", "1", "-ar", "16000", str(out_path)]
    _run(cmd)
    return out_path


def download_audio(
    youtube_url: str,
    out_dir: str = "runs/audio",
    cookies_from_browser: Optional[str] = None,  # "opera" / "brave" / "chrome" / "edge"
) -> Path:
    """
    Downloads best audio from YouTube and produces a single normalized file:
    out_dir/audio_16k.wav

    Requires: yt-dlp and ffmpeg in PATH.
    """
    _ensure_tool("yt-dlp")
    _ensure_tool("ffmpeg")

    outp = Path(out_dir)
    outp.mkdir(parents=True, exist_ok=True)

    # Download audio-only (keep original container), then normalize ourselves.
    temp_template = outp / "audio_source.%(ext)s"
    cmd = [
        "yt-dlp",
        "-f", "bestaudio/best",
        "-o", str(temp_template),
        youtube_url,
    ]
    if cookies_from_browser:
        cmd += ["--cookies-from-browser", cookies_from_browser]

    _run(cmd)

    # Find the produced downloaded file (any audio container)
    candidates = []
    for p in outp.iterdir():
        if p.is_file() and p.stem.startswith("audio_source") and p.suffix.lower() in AUDIO_EXTS:
            candidates.append(p)

    if not candidates:
        # fallback: any audio file in folder
        candidates = [p for p in outp.iterdir() if p.is_file() and p.suffix.lower() in AUDIO_EXTS]

    if not candidates:
        raise FileNotFoundError(
            "yt-dlp did not produce an audio file. Check yt-dlp output and ensure ffmpeg works."
        )

    src = sorted(candidates, key=lambda x: x.stat().st_size, reverse=True)[0]

    wav16k = outp / "audio_16k.wav"
    normalize_to_wav16k(src, wav16k)
    return wav16k


def transcribe_whisper(
    audio_path: str | Path,
    model_size: str = "base",
    out_dir: Optional[str | Path] = None,
) -> Dict[str, Any]:
    """
    Speech-to-text with timestamps using openai-whisper.
    Accepts any audio format; normalizes to 16k mono wav first.
    """
    import whisper  # lazy import

    audio_path = Path(audio_path)

    # Where to store normalized wav
    if out_dir is None:
        # If input is inside "lecture XX", keep normalized audio in that lecture's outputs/
        # otherwise store alongside the input.
        guess = audio_path.parent / "outputs"
        outp = guess
    else:
        outp = Path(out_dir)

    outp.mkdir(parents=True, exist_ok=True)
    wav16k = outp / "audio_16k.wav"

    # If input already wav, still normalize to be safe (sample rate/channel issues happen)
    if not wav16k.exists():
        normalize_to_wav16k(audio_path, wav16k)

    model = whisper.load_model(model_size)
    result = model.transcribe(str(wav16k), fp16=False, verbose=False)

    segments = []
    for s in result.get("segments", []):
        segments.append({
            "start": float(s["start"]),
            "end": float(s["end"]),
            "text": (s.get("text") or "").strip()
        })

    return {
        "language": result.get("language"),
        "text": (result.get("text") or "").strip(),
        "segments": segments,
        "audio_wav16k": str(wav16k),
    }