"""DOCX redaction via run-level offset mapping.

Maps paragraph-text offsets back to the individual runs that make them up
and blacks out only the overlapping portion of each run, so an entity that
straddles two runs (e.g. bold+plain) still redacts correctly and per-run
formatting survives.

Coverage: body paragraphs, tables (including nested tables and merged
cells), and every header/footer variant. Earlier versions only walked
`document.paragraphs`, which excludes all of those - measured on form-style
documents, 66% of PII stayed in the output even when every entity was
correctly selected, because forms keep their fields in tables and the
letterhead in the header. Document properties (author, last-modified-by,
title...) are scrubbed too, since they routinely carry the author's name.
"""

import logging
from typing import Iterator, List, Tuple

from docx import Document as DocxDocument

from app.redaction.matching import find_matches, selected_texts

logger = logging.getLogger("blacken")

REDACTION_CHAR = "█"

_CORE_PROPERTY_FIELDS = (
    "author", "last_modified_by", "title", "subject", "keywords",
    "comments", "category", "content_status", "identifier",
)


def _iter_block_paragraphs(container) -> Iterator:
    """Paragraphs of a body/cell/header, descending into tables. Merged cells
    appear once per spanned grid column in python-docx, so dedupe them.

    Dedupe on the lxml elements themselves, never on id(): lxml proxies are
    garbage-collected between rows and their ids get reused, which made
    later rows look "already seen" - they were skipped and leaked (caught by
    evals/run_e2e.py's oracle mode). Holding the elements in the set keeps
    them alive and compares by identity."""
    yield from container.paragraphs
    seen_cells = set()
    for table in getattr(container, "tables", []):
        for row in table.rows:
            for cell in row.cells:
                if cell._tc in seen_cells:
                    continue
                seen_cells.add(cell._tc)
                yield from _iter_block_paragraphs(cell)


def iter_docx_paragraphs(doc) -> Iterator:
    """Every paragraph a reader can see: headers first (top of the page),
    then the body, then footers. Shared with text extraction so detection
    and redaction always cover exactly the same text."""
    seen_parts = set()
    headers, footers = [], []
    for section in doc.sections:
        for attr in ("first_page_header", "header", "even_page_header"):
            headers.append(getattr(section, attr))
        for attr in ("first_page_footer", "footer", "even_page_footer"):
            footers.append(getattr(section, attr))

    def unique(parts):
        for part in parts:
            # Linked-to-previous headers have no content of their own.
            if part.is_linked_to_previous or part._element in seen_parts:
                continue
            seen_parts.add(part._element)
            yield part

    for header in unique(headers):
        yield from _iter_block_paragraphs(header)
    yield from _iter_block_paragraphs(doc)
    for footer in unique(footers):
        yield from _iter_block_paragraphs(footer)


def paragraph_runs(paragraph) -> list:
    """Runs in reading order *including* those inside hyperlinks.
    `paragraph.runs` skips hyperlink runs, and in real Word files email
    addresses and profile URLs are almost always hyperlinks - so they were
    neither detected nor redacted."""
    runs = []
    for item in paragraph.iter_inner_content():
        if hasattr(item, "runs"):  # Hyperlink
            runs.extend(item.runs)
        else:
            runs.append(item)
    return runs


def paragraph_text(paragraph) -> str:
    return "".join(run.text for run in paragraph_runs(paragraph))


def _paragraph_text_and_run_spans(paragraph) -> Tuple[str, List[Tuple[int, int, object]]]:
    spans = []
    offset = 0
    runs = paragraph_runs(paragraph)
    for run in runs:
        start = offset
        offset += len(run.text)
        spans.append((start, offset, run))
    return "".join(run.text for run in runs), spans


def _scrub_core_properties(doc) -> None:
    props = doc.core_properties
    for name in _CORE_PROPERTY_FIELDS:
        try:
            setattr(props, name, "")
        except (AttributeError, ValueError):
            continue


def redact_docx(original_path: str, entities: List[dict], output_path: str) -> int:
    needles = selected_texts(entities)
    doc = DocxDocument(original_path)
    total_redactions = 0

    for paragraph in iter_docx_paragraphs(doc):
        para_text, run_spans = _paragraph_text_and_run_spans(paragraph)
        if not para_text:
            continue

        for match_start, match_end in find_matches(para_text, needles):
            for run_start, run_end, run in run_spans:
                overlap_start = max(match_start, run_start)
                overlap_end = min(match_end, run_end)
                if overlap_start >= overlap_end:
                    continue
                local_start = overlap_start - run_start
                local_end = overlap_end - run_start
                text = run.text
                run.text = (
                    text[:local_start]
                    + REDACTION_CHAR * (local_end - local_start)
                    + text[local_end:]
                )
                total_redactions += 1

    _scrub_core_properties(doc)
    doc.save(output_path)
    logger.info("DOCX redaction: %d run segment(s) redacted", total_redactions)
    return total_redactions
