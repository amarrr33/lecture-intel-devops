from __future__ import annotations
import os
import subprocess
from pathlib import Path
from typing import Optional, Dict, Any

def _run(cmd: list[str]) -> None:
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"Command failed:\n{cmd}\nSTDERR:\n{p.stderr}")

def download_audio(youtube_url: str, out_dir: str = "runs/audio") -> Path:
    """
    Downloads best audio and converts to WAV 16k mono using ffmpeg (via yt-dlp postprocessor).
    Requires: ffmpeg in PATH.
    """
    outp = Path(out_dir)
    outp.mkdir(parents=True, exist_ok=True)
    wav_path = outp / "audio.wav"
    # yt-dlp can extract audio; we still normalize via ffmpeg to 16k wav for whisper stability
    temp = outp / "audio_source.%(ext)s"
    cmd = [
        "yt-dlp",
        "-f", "bestaudio/best",
        "-o", str(temp),
        "--extract-audio",
        "--audio-format", "wav",
        "--audio-quality", "0",
        youtube_url,
    ]
    _run(cmd)

    # find produced wav (yt-dlp names it audio_source.wav)
    produced = list(outp.glob("audio_source.wav"))
    if not produced:
        # fallback: any wav in folder
        produced = list(outp.glob("*.wav"))
    if not produced:
        raise FileNotFoundError("yt-dlp did not produce a wav file. Is ffmpeg installed?")
    src = produced[0]

    # normalize to 16k mono wav (Whisper friendly)
    cmd2 = [
        "ffmpeg", "-y",
        "-i", str(src),
        "-ac", "1",
        "-ar", "16000",
        str(wav_path)
    ]
    _run(cmd2)
    return wav_path

def transcribe_whisper(audio_path: str | Path, model_size: str = "base") -> Dict[str, Any]:
    """
    Speech-to-text with timestamps using openai-whisper.
    """
    import whisper  # lazy import

    audio_path = str(audio_path)
    model = whisper.load_model(model_size)
    result = model.transcribe(audio_path, fp16=False, verbose=False)

    # Return segments: [{start,end,text}]
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
    }
