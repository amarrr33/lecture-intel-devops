from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List
import re

AUDIO_EXTS = {".wav", ".mp3", ".webm", ".m4a", ".aac", ".flac", ".ogg"}
SLIDE_EXTS = {".pdf", ".pptx"}

# More flexible patterns
LECTURE_RE = re.compile(r"lecture[\s_\-]*(\d{1,3})", re.IGNORECASE)

@dataclass
class LectureFiles:
    lecture_num: int
    folder: Path
    audio: Optional[Path]
    slides: Optional[Path]

# ---------------------------
# Helpers
# ---------------------------

def parse_lecture_num(folder_name: str) -> Optional[int]:
    """
    Handles:
    lecture 1
    Lecture_01
    lecture-02
    lecture03
    """
    m = LECTURE_RE.search(folder_name)
    return int(m.group(1)) if m else None

def pick_largest(files: List[Path]) -> Optional[Path]:
    if not files:
        return None
    return max(files, key=lambda p: p.stat().st_size)

# ---------------------------
# File detection (SMART)
# ---------------------------

def find_audio(folder: Path, lecture_num: int) -> Optional[Path]:
    files = [p for p in folder.iterdir() if p.is_file()]

    auds = [p for p in files if p.suffix.lower() in AUDIO_EXTS]
    if not auds:
        return None

    # Prefer matching lecture number
    prefix = f"{lecture_num:02d}"
    for p in auds:
        if prefix in p.stem:
            return p

    # fallback → largest audio
    return pick_largest(auds)

def find_slides(folder: Path) -> Optional[Path]:
    files = [p for p in folder.iterdir() if p.is_file()]
    slides = [p for p in files if p.suffix.lower() in SLIDE_EXTS]

    if not slides:
        return None

    # Prefer PDFs (usually better structured)
    pdfs = [p for p in slides if p.suffix.lower() == ".pdf"]
    if pdfs:
        return pick_largest(pdfs)

    return pick_largest(slides)

# ---------------------------
# Main scanner
# ---------------------------

def scan_dataset(data_dir: str | Path) -> List[LectureFiles]:
    data_dir = Path(data_dir)

    if not data_dir.exists():
        raise FileNotFoundError(f"{data_dir} does not exist")

    lectures: List[LectureFiles] = []

    for d in sorted(data_dir.iterdir()):
        if not d.is_dir():
            continue

        num = parse_lecture_num(d.name)
        if num is None:
            continue

        audio = find_audio(d, num)
        slides = find_slides(d)

        # 🚨 Skip useless entries
        if not audio and not slides:
            print(f"⚠️ Skipping {d.name} (no audio & no slides)")
            continue

        lectures.append(
            LectureFiles(
                lecture_num=num,
                folder=d,
                audio=audio,
                slides=slides,
            )
        )

    lectures.sort(key=lambda x: x.lecture_num)

    print(f"✅ Found {len(lectures)} valid lectures")
    return lectures

# ---------------------------
# Output folder
# ---------------------------

def outputs_dir(lecture_folder: Path) -> Path:
    od = lecture_folder / "outputs"
    od.mkdir(parents=True, exist_ok=True)
    return od