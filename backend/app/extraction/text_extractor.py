"""Extract plain text from uploaded documents for entity detection.

PDF pages with little or no extractable text (scanned/image-only pages)
fall back to OCR automatically, instead of silently returning near-empty
text the way the previous implementation did.
"""

import asyncio
import logging
from pathlib import Path

import aiofiles
import pymupdf as fitz  # PyMuPDF; `fitz` is the deprecated import alias
from docx import Document as DocxDocument
from PIL import Image

from app.redaction.image_redactor import ocr_extract_text

logger = logging.getLogger("blacken")

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tiff", ".bmp"}

# A page with fewer non-whitespace characters than this is treated as
# having no usable text layer and gets OCR'd instead.
MIN_TEXT_CHARS_PER_PAGE = 20


class ExtractionError(Exception):
    """Raised when a document's text can't be extracted."""


def pixmap_to_pil(pix: "fitz.Pixmap") -> Image.Image:
    if pix.n >= 4:  # has alpha or CMYK - normalize to RGB
        pix = fitz.Pixmap(fitz.csRGB, pix)
    return Image.frombytes("RGB", (pix.width, pix.height), pix.samples)


def _ocr_pdf_page(page: "fitz.Page", dpi: int | None = None) -> str:
    from app.config import settings

    # Same DPI as the scanned-page redactor (settings.ocr_dpi) so OCR text at
    # redaction time matches what detection saw.
    pix = page.get_pixmap(dpi=dpi or settings.ocr_dpi)
    image = pixmap_to_pil(pix)
    return ocr_extract_text(image)


def extract_text_from_pdf(file_path: str) -> str:
    text_parts = []
    with fitz.open(file_path) as doc:
        for page in doc:
            page_text = page.get_text()
            if len(page_text.strip()) < MIN_TEXT_CHARS_PER_PAGE:
                logger.info("Page has little/no text layer, falling back to OCR")
                page_text = _ocr_pdf_page(page)
            text_parts.append(page_text)
    return "\n".join(text_parts)


def extract_text_from_docx(file_path: str) -> str:
    # Same traversal the redactor uses (body + tables + headers/footers):
    # text that detection never sees can never be redacted.
    from app.redaction.docx_redactor import iter_docx_paragraphs, paragraph_text

    doc = DocxDocument(file_path)
    return "\n".join(paragraph_text(p) for p in iter_docx_paragraphs(doc))


def extract_text_from_image(file_path: str) -> str:
    image = Image.open(file_path).convert("RGB")
    return ocr_extract_text(image)


def _extract_text_sync(file_path: str, filename: str) -> str:
    ext = Path(filename).suffix.lower()
    try:
        if ext == ".pdf":
            return extract_text_from_pdf(file_path)
        if ext in {".docx", ".doc"}:
            return extract_text_from_docx(file_path)
        if ext in IMAGE_EXTENSIONS:
            return extract_text_from_image(file_path)
    except Exception as exc:  # noqa: BLE001 - surfaced as ExtractionError
        raise ExtractionError(f"Failed to process {ext} file: {exc}") from exc

    # Plain text fallback
    for encoding in ("utf-8", "latin-1"):
        try:
            with open(file_path, "r", encoding=encoding) as f:
                return f.read()
        except UnicodeDecodeError:
            continue
    raise ExtractionError(f"Unsupported or unreadable file type: {ext}")


async def extract_text(file_path: str, filename: str) -> str:
    """Extract text based on file extension, off the event loop."""
    return await asyncio.to_thread(_extract_text_sync, file_path, filename)
