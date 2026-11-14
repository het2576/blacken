"""PDF redaction via word/bbox offset mapping.

The previous implementation re-searched the page for each entity's
literal text via `page.search_for(text)`. That's fragile: it depends on
exact substring/case matches against however PyMuPDF happens to expose
searchable text, and treats each entity independently with no shared
understanding of the page layout.

This version reconstructs each page's text once from PyMuPDF's per-word
bounding boxes, keeping an offset->bbox map, then locates every entity as
a substring of that reconstructed text and redacts the union of word
boxes it overlaps - which correctly handles multi-word and wrapped
matches. Pages with no real text layer (scanned pages) are rasterized,
OCR'd and redacted with the same line-reconstruction approach used for
image files, then the page content is replaced with the redacted image.
"""

import io
import logging
from typing import List, Tuple

import pymupdf as fitz  # PyMuPDF; `fitz` is the deprecated import alias

from app.extraction.text_extractor import (
    MIN_TEXT_CHARS_PER_PAGE,
    pixmap_to_pil,
)
from app.redaction.image_redactor import draw_boxes, find_redaction_boxes
from app.redaction.matching import find_matches, selected_texts

logger = logging.getLogger("blacken")

Bbox = Tuple[float, float, float, float]
Span = Tuple[int, int, Bbox]



def _page_text_and_spans(page: "fitz.Page") -> Tuple[str, List[Span]]:
    """Reconstruct a page's text from its words, in reading order, with an
    offset->bbox map so character spans can be mapped back to boxes."""
    words = page.get_text("words")  # x0,y0,x1,y1,word,block_no,line_no,word_no
    words.sort(key=lambda w: (w[5], w[6], w[7]))

    text_parts: List[str] = []
    spans: List[Span] = []
    offset = 0
    last_block_line = None

    for x0, y0, x1, y1, word, block_no, line_no, _word_no in words:
        block_line = (block_no, line_no)
        if last_block_line is not None:
            separator = "\n" if block_line != last_block_line else " "
            text_parts.append(separator)
            offset += len(separator)
        start = offset
        text_parts.append(word)
        offset += len(word)
        spans.append((start, offset, (x0, y0, x1, y1)))
        last_block_line = block_line

    return "".join(text_parts), spans


def _boxes_for_page_text(
    page_text: str, spans: List[Span], entities: List[dict]
) -> List[Bbox]:
    boxes: List[Bbox] = []
    for match_start, match_end in find_matches(page_text, selected_texts(entities)):
        boxes.extend(bbox for (s, e, bbox) in spans if s < match_end and e > match_start)
    return boxes


# Word boxes from PyMuPDF span the font's full ascender-to-descender height,
# which on tightly leaded text overlaps the glyph boxes of the lines above
# and below. apply_redactions() deletes every character whose box touches
# the redaction rect, so full-height rects silently deleted words on
# neighbouring lines. Removal uses a vertically inset rect (still covering
# the middle of every target glyph); the visible black bar is drawn at full
# height afterwards as plain vector content, which removes nothing.
_REMOVAL_INSET = 0.3


def _removal_rect(bbox: Bbox) -> "fitz.Rect":
    x0, y0, x1, y1 = bbox
    inset = (y1 - y0) * _REMOVAL_INSET
    return fitz.Rect(x0, y0 + inset, x1, y1 - inset)


def _scrub_metadata(doc: "fitz.Document") -> None:
    """Author/title/subject/keywords and XMP metadata routinely carry the
    very names being redacted from the page content."""
    doc.set_metadata({})
    try:
        doc.del_xml_metadata()
    except Exception:  # noqa: BLE001 - absent XMP is fine
        pass


def _redact_scanned_page(page: "fitz.Page", entities: List[dict]) -> int:
    """Rasterize a page with no usable text layer, OCR + redact it as an
    image, and replace the page's content with the redacted rasterization."""
    from app.config import settings

    # Must equal the extraction DPI so OCR text matches what detection saw.
    pix = page.get_pixmap(dpi=settings.ocr_dpi)
    image = pixmap_to_pil(pix)

    boxes = find_redaction_boxes(image, entities)
    if not boxes:
        return 0

    # OCR boxes are in raster pixel space; scale back to PDF point space.
    scale_x = page.rect.width / pix.width
    scale_y = page.rect.height / pix.height
    pdf_boxes = [
        (x0 * scale_x, y0 * scale_y, x1 * scale_x, y1 * scale_y) for x0, y0, x1, y1 in boxes
    ]

    draw_boxes(image, boxes)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")

    rect = page.rect
    # Clear the page's existing content, then drop in the redacted image.
    page.add_redact_annot(rect, fill=(1, 1, 1))
    page.apply_redactions()
    page.insert_image(rect, stream=buffer.getvalue())

    return len(pdf_boxes)


def redact_pdf(original_path: str, entities: List[dict], output_path: str) -> int:
    """Always writes `output_path` - even with nothing selected the output is
    a metadata-scrubbed copy. (Returning early without saving made the API
    answer 500 "Redaction produced no output file" for any document where the
    user deselected everything or nothing was auto-selected.)"""
    selected = [e for e in entities if e.get("selected", True) and e.get("text", "").strip()]

    total_redactions = 0
    doc = fitz.open(original_path)
    try:
        for page in doc if selected else []:
            page_text, spans = _page_text_and_spans(page)

            if len(page_text.strip()) < MIN_TEXT_CHARS_PER_PAGE:
                total_redactions += _redact_scanned_page(page, selected)
                continue

            boxes = _boxes_for_page_text(page_text, spans, selected)
            for bbox in boxes:
                page.add_redact_annot(_removal_rect(bbox), fill=False)
            if boxes:
                page.apply_redactions()
                for bbox in boxes:
                    page.draw_rect(fitz.Rect(bbox), color=(0, 0, 0), fill=(0, 0, 0), width=0)
            total_redactions += len(boxes)

        _scrub_metadata(doc)
        doc.save(output_path, garbage=4, deflate=True, clean=True)
    finally:
        doc.close()

    logger.info("PDF redaction: %d region(s) redacted", total_redactions)
    return total_redactions
