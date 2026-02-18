from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
import re

@dataclass
class SlideText:
    slide_id: str
    text: str

def _clean(t: str) -> str:
    t = re.sub(r"\s+", " ", t).strip()
    return t

def extract_from_pptx(pptx_path: str | Path) -> List[SlideText]:
    from pptx import Presentation
    prs = Presentation(str(pptx_path))
    out: List[SlideText] = []
    for i, slide in enumerate(prs.slides, start=1):
        chunks = []
        for shape in slide.shapes:
            if hasattr(shape, "text") and shape.text:
                chunks.append(shape.text)
        text = _clean("\n".join(chunks))
        out.append(SlideText(slide_id=f"slide_{i}", text=text))
    return out

def extract_from_pdf_text(pdf_path: str | Path) -> List[SlideText]:
    import pdfplumber
    out: List[SlideText] = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            out.append(SlideText(slide_id=f"page_{i}", text=_clean(text)))
    return out

def ocr_pdf_pages(pdf_path: str | Path) -> List[SlideText]:
    """
    Optional OCR path. Needs:
      - Tesseract installed
      - poppler installed (pdfplumber uses pdf2image-like rendering via pdfplumber? If OCR is needed,
        simplest is to avoid OCR for PDF in demo OR use screenshots externally.)
    This function tries a very basic rasterize->OCR using pdfplumber page.to_image().
    """
    import pdfplumber
    from PIL import Image
    import pytesseract

    out: List[SlideText] = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            try:
                im = page.to_image(resolution=200).original  # PIL Image
                text = pytesseract.image_to_string(im)
            except Exception:
                text = ""
            out.append(SlideText(slide_id=f"page_{i}", text=_clean(text)))
    return out

def load_slides(file_path: str | Path, use_ocr_if_empty: bool = False) -> List[SlideText]:
    p = Path(file_path)
    if not p.exists():
        raise FileNotFoundError(p)

    if p.suffix.lower() == ".pptx":
        return extract_from_pptx(p)

    if p.suffix.lower() == ".pdf":
        slides = extract_from_pdf_text(p)
        if use_ocr_if_empty and sum(1 for s in slides if s.text) < max(1, len(slides)//4):
            # if most pages are empty text, try OCR
            return ocr_pdf_pages(p)
        return slides

    raise ValueError("Only .pptx or .pdf supported in this minimal build.")
