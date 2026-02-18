from __future__ import annotations
from pathlib import Path
import json
from typing import Dict, Any

def save_json(obj: Dict[str, Any], out_path: str | Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")

def save_markdown(alignment: Dict[str, Any], out_path: str | Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    lines = ["# Slide ↔ Speech Alignment Report\n"]
    stats = alignment.get("stats", {})
    lines.append(f"- Slides: **{stats.get('num_slides')}**")
    lines.append(f"- Transcript segments: **{stats.get('num_segments')}**")
    lines.append(f"- Avg best similarity: **{stats.get('avg_best_score'):.3f}**\n")

    for a in alignment.get("alignments", []):
        seg = a["segment"]
        lines.append(f"## {a['slide_id']}")
        lines.append(f"- Best segment: `{a['best_segment_idx']}`")
        lines.append(f"- Score: **{a['score']:.3f}**")
        lines.append(f"- Time: **{seg['start']:.1f}s → {seg['end']:.1f}s**")
        lines.append(f"- Text: {seg['text']}\n")

    out_path.write_text("\n".join(lines), encoding="utf-8")
