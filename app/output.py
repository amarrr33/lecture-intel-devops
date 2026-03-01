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
    avg_conf = stats.get('avg_confidence', 0.0)
    lines.append(f"- Avg confidence: **{avg_conf:.3f}**\n")

    for a in alignment.get("alignments", []):
        seg = a.get("segment", {})
        lines.append(f"## {a.get('slide_id', 'unknown')}")
        lines.append(f"- Best segment: `{a.get('best_segment_idx')}`")
        lines.append(f"- Cosine: **{a.get('cosine', 0.0):.3f}**")
        lines.append(f"- Confidence: **{a.get('confidence', 0.0):.3f}**")
        lines.append(f"- Time: **{seg.get('start', 0.0):.1f}s → {seg.get('end', 0.0):.1f}s**")
        lines.append(f"- Text: {seg.get('text', '')}\n")

    out_path.write_text("\n".join(lines), encoding="utf-8")
