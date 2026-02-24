from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, List
import re

AUDIO_EXTS = {".wav", ".mp3", ".webm", ".m4a", ".aac", ".flac", ".ogg"}
SLIDE_EXTS = {".pdf", ".pptx"}

LECTURE_RE = re.compile(r"^lecture\s+(\d{1,3})$", re.IGNORECASE)

@dataclass
class LectureFiles:
    lecture_num: int
    folder: Path
    audio: Optional[Path]
    slides: Optional[Path]

def parse_lecture_num(folder_name: str) -> Optional[int]:
    m = LECTURE_RE.match(folder_name.strip())
    return int(m.group(1)) if m else None

def pick_largest(files: List[Path]) -> Optional[Path]:
    if not files:
        return None
    return max(files, key=lambda p: p.stat().st_size)

def find_audio(folder: Path, lecture_num: int) -> Optional[Path]:
    # Prefer lecture03.* naming
    prefix = f"lecture{lecture_num:02d}".lower()
    auds = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in AUDIO_EXTS]
    for p in sorted(auds):
        if p.stem.lower().startswith(prefix):
            return p
    # fallback: any audio file
    return sorted(auds)[0] if auds else None

def find_slides(folder: Path) -> Optional[Path]:
    # Slides have random names → choose largest pdf/pptx in folder
    slides = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in SLIDE_EXTS]
    return pick_largest(slides)

def scan_dataset(data_dir: str | Path) -> List[LectureFiles]:
    data_dir = Path(data_dir)
    out: List[LectureFiles] = []
    for d in data_dir.iterdir():
        if not d.is_dir():
            continue
        n = parse_lecture_num(d.name)
        if n is None:
            continue
        out.append(LectureFiles(
            lecture_num=n,
            folder=d,
            audio=find_audio(d, n),
            slides=find_slides(d),
        ))
    out.sort(key=lambda x: x.lecture_num)
    return out

def outputs_dir(lecture_folder: Path) -> Path:
    od = lecture_folder / "outputs"
    od.mkdir(exist_ok=True)
    return od