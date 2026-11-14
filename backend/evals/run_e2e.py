"""End-to-end redaction evaluation through the real HTTP API.

    python -m evals.run_e2e                    # all formats, gold docs
    python -m evals.run_e2e --formats pdf docx --ai4privacy 100

For every annotated document and every file format, this:
  1. renders the document into a real file (PDF text layer, scanned PDF,
     DOCX, DOCX with PII in tables/headers, PNG),
  2. drives upload -> redact -> download through FastAPI exactly as the
     frontend does,
  3. extracts every recoverable string from the *output file* - PDF text
     layer, all DOCX XML parts (body, tables, headers, footers, document
     properties), OCR for images - and checks which gold PII values survived.

Metrics per format:
  leak_rate_auto    - share of gold PII values still recoverable after the
                      default one-click redaction (detection + redaction).
  leak_rate_oracle  - same, but redacting the *gold* entities: isolates the
                      redaction/rendering layer from detection. Should be ~0;
                      anything else is a redactor bug.
  over_redaction    - share of non-PII words destroyed (collateral damage).
  metadata_leak     - output file still carries the author's name in its
                      document properties.

For OCR'd formats a value only counts if it was readable in the
*unredacted* rendering, so OCR misreads can't masquerade as redactions.
"""

import argparse
import io
import json
import re
import statistics
import time
import uuid
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

from evals import datasets as ds
from evals.taxonomy import CORE_CLASSES

RESULTS_DIR = Path(__file__).resolve().parent / "results"
FORMATS = ("pdf", "pdf_scanned", "docx", "docx_structured", "png")
AUTHOR = "Evaluation Author Rosalind Featherstone"  # canary for metadata leaks
_WORD = re.compile(r"[A-Za-z0-9]+")
_FONT_CANDIDATES = (
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/Library/Fonts/Arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/dejavu/DejaVuSans.ttf",
)


# --------------------------------------------------------------- rendering
def _wrap(text: str, width: int):
    import textwrap

    out = []
    for line in text.split("\n"):
        out.extend(textwrap.wrap(line, width) or [""])
    return out


def build_pdf(text: str) -> bytes:
    import pymupdf

    doc = pymupdf.open()
    lines = _wrap(text, 95)
    for i in range(0, len(lines), 60):
        page = doc.new_page()
        y = 60
        for line in lines[i : i + 60]:
            page.insert_text((50, y), line, fontsize=10)
            y += 12
    doc.set_metadata({"author": AUTHOR, "title": "Evaluation document"})
    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


def _render_image(text: str):
    from PIL import Image, ImageDraw, ImageFont

    font_path = next((p for p in _FONT_CANDIDATES if Path(p).exists()), None)
    font = ImageFont.truetype(font_path, 30) if font_path else ImageFont.load_default()
    lines = _wrap(text, 80)
    img = Image.new("RGB", (1700, 80 + 42 * len(lines)), "white")
    draw = ImageDraw.Draw(img)
    for i, line in enumerate(lines):
        draw.text((40, 40 + 42 * i), line, fill="black", font=font)
    return img


def build_png(text: str) -> bytes:
    buf = io.BytesIO()
    _render_image(text).save(buf, format="PNG")
    return buf.getvalue()


def build_scanned_pdf(text: str) -> bytes:
    import pymupdf

    png = build_png(text)
    doc = pymupdf.open()
    img = pymupdf.open(stream=png, filetype="png")
    rect = img[0].rect
    page = doc.new_page(width=rect.width * 0.5, height=rect.height * 0.5)
    page.insert_image(page.rect, stream=png)
    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


def build_docx(text: str, structured: bool = False) -> bytes:
    """structured=True moves the first line into the page header and every
    `Label: value` line into a two-column table - how real forms are built."""
    from docx import Document

    doc = Document()
    doc.core_properties.author = AUTHOR
    lines = text.split("\n")
    if structured and lines:
        doc.sections[0].header.paragraphs[0].text = lines[0]
        lines = lines[1:]
        kv = [l for l in lines if re.match(r"^[A-Za-z][A-Za-z '/.()-]{1,40}:\s+\S", l)]
        rest = [l for l in lines if l not in kv]
        if kv:
            table = doc.add_table(rows=0, cols=2)
            for l in kv:
                k, v = l.split(":", 1)
                row = table.add_row().cells
                row[0].text, row[1].text = k + ":", v.strip()
        lines = rest
    for line in lines:
        doc.add_paragraph(line)
    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


BUILDERS = {
    "pdf": (build_pdf, ".pdf"),
    "pdf_scanned": (build_scanned_pdf, ".pdf"),
    "docx": (build_docx, ".docx"),
    "docx_structured": (lambda t: build_docx(t, structured=True), ".docx"),
    "png": (build_png, ".png"),
}


# -------------------------------------------------------------- extraction
def _ocr(img) -> str:
    import pytesseract

    return pytesseract.image_to_string(img)


def recoverable_text(data: bytes, fmt: str, include_metadata: bool = True) -> str:
    """Everything an adversary could read back out of the output file.
    `include_metadata=False` gives only visible content - used for
    over-redaction, where scrubbed metadata must not count as lost text."""
    if fmt in ("pdf", "pdf_scanned"):
        import pymupdf

        doc = pymupdf.open(stream=data, filetype="pdf")
        parts = [page.get_text() for page in doc]
        if include_metadata:
            parts.append(json.dumps(doc.metadata or {}))
        if fmt == "pdf_scanned":
            from app.extraction.text_extractor import pixmap_to_pil

            parts.extend(_ocr(pixmap_to_pil(p.get_pixmap(dpi=200))) for p in doc)
        return "\n".join(parts)
    if fmt.startswith("docx"):
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            xml = " ".join(
                z.read(n).decode("utf-8", "ignore")
                for n in z.namelist()
                if n.endswith(".xml") and (include_metadata or not n.startswith("docProps/"))
            )
        # Paragraph/cell boundaries become newlines; runs within a paragraph
        # join with no separator, exactly as Word displays them.
        xml = re.sub(r"</w:p>|</w:tc>|</w:tr>", "\n", xml)
        return re.sub(r"<[^>]+>", "", xml)
    if fmt == "png":
        from PIL import Image

        return _ocr(Image.open(io.BytesIO(data)))
    raise ValueError(fmt)


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


def _contains(haystack_norm: str, value: str) -> bool:
    v = _norm(value)
    return len(v) >= 2 and v in haystack_norm


# -------------------------------------------------------------- evaluation
def gold_values(sample):
    return [
        (sample.text[s:e], cls) for s, e, cls in sample.spans if cls in CORE_CLASSES
    ]


def run(client, samples, formats, owner_id):
    headers = {"X-Owner-Id": owner_id}
    per_format = {}
    for fmt in formats:
        build, ext = BUILDERS[fmt]
        stats = Counter()
        by_class = defaultdict(Counter)
        leaks = Counter()
        latencies = []
        for s in samples:
            golds = gold_values(s)
            if not golds:
                continue
            data = build(s.text)
            original_view = _norm(recoverable_text(data, fmt))
            observable = [(v, c) for v, c in golds if _contains(original_view, v)]
            stats["gold_values"] += len(golds)
            stats["observable"] += len(observable)

            # Words inside any annotated span - core PII *and* don't-care
            # IGNORE/ORGANIZATION regions - are excluded, consistent with the
            # detection metrics; what's left is text that must survive.
            annotated = [s.text[a:b] for a, b, _c in s.spans]
            original_content = _norm(recoverable_text(data, fmt, include_metadata=False))
            non_pii_words = Counter(
                w.lower() for w in _WORD.findall(original_content)
            ) - Counter(w.lower() for v in annotated for w in _WORD.findall(v))

            for mode in ("auto", "oracle"):
                t0 = time.perf_counter()
                up = client.post(
                    "/api/upload",
                    files={"file": (f"{s.id}{ext}", data, "application/octet-stream")},
                    headers=headers,
                )
                if up.status_code != 200:
                    stats[f"{mode}_errors"] += 1
                    stats[f"{mode}_errors_http_{up.status_code}"] += 1
                    continue
                doc_id = up.json()["document_id"]
                body = {"document_id": doc_id}
                if mode == "oracle":
                    body["entities"] = [
                        {"text": v, "type": c, "start": 0, "end": len(v), "selected": True}
                        for v, c in golds
                    ]
                rr = client.post("/api/redact", json=body, headers=headers)
                if rr.status_code != 200:
                    stats[f"{mode}_errors"] += 1
                    client.delete(f"/api/document/{doc_id}", headers=headers)
                    continue
                out = client.get(f"/api/download/{doc_id}", headers=headers).content
                if mode == "auto":
                    latencies.append((time.perf_counter() - t0) * 1000)
                client.delete(f"/api/document/{doc_id}", headers=headers)

                out_view = _norm(recoverable_text(out, fmt))
                for v, c in observable:
                    leaked = _contains(out_view, v)
                    stats[f"{mode}_leaked"] += leaked
                    by_class[c][f"{mode}_total"] += 1
                    by_class[c][f"{mode}_leaked"] += leaked
                    if leaked and mode == "auto":
                        leaks[(c, v)] += 1
                stats[f"{mode}_docs"] += 1
                if AUTHOR.lower() in out_view:
                    stats[f"{mode}_metadata_leak"] += 1
                if mode == "auto":
                    out_content = _norm(recoverable_text(out, fmt, include_metadata=False))
                    out_words = Counter(w.lower() for w in _WORD.findall(out_content))
                    lost = sum((non_pii_words - out_words).values())
                    stats["non_pii_words"] += sum(non_pii_words.values())
                    stats["non_pii_lost"] += lost

        obs = stats["observable"] or 1
        per_format[fmt] = {
            "documents": stats["auto_docs"],
            "gold_values": stats["gold_values"],
            "observable_values": stats["observable"],
            "leak_rate_auto": stats["auto_leaked"] / obs,
            "leak_rate_oracle": stats["oracle_leaked"] / obs,
            "redaction_recall_auto": 1 - stats["auto_leaked"] / obs,
            "over_redaction_rate": stats["non_pii_lost"] / (stats["non_pii_words"] or 1),
            "metadata_leak_docs": stats["auto_metadata_leak"],
            "errors": {k: v for k, v in stats.items() if k.endswith("_errors")},
            "latency_ms_p50": statistics.median(latencies) if latencies else None,
            "latency_ms_p95": sorted(latencies)[int(0.95 * (len(latencies) - 1))] if latencies else None,
            "leak_rate_by_class_auto": {
                c: v["auto_leaked"] / v["auto_total"] for c, v in sorted(by_class.items()) if v["auto_total"]
            },
            "leak_rate_by_class_oracle": {
                c: v["oracle_leaked"] / v["oracle_total"] for c, v in sorted(by_class.items()) if v["oracle_total"]
            },
            "top_leaks": [[c, v, n] for (c, v), n in leaks.most_common(15)],
        }
        r = per_format[fmt]
        print(
            f"{fmt:16} docs={r['documents']:3}  observable={r['observable_values']:4}  "
            f"leak(auto)={r['leak_rate_auto']*100:5.1f}%  leak(oracle)={r['leak_rate_oracle']*100:5.1f}%  "
            f"over-redact={r['over_redaction_rate']*100:4.1f}%  meta-leak={r['metadata_leak_docs']}  "
            f"p50={r['latency_ms_p50'] or 0:.0f}ms  errors={r['errors']}"
        )
    return per_format


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--formats", nargs="+", default=list(FORMATS), choices=FORMATS)
    ap.add_argument("--ai4privacy", type=int, default=0, help="also include N ai4privacy test docs")
    ap.add_argument("--tag", default="")
    ap.add_argument("--no-calibration", action="store_true", help="ignore app/detection/thresholds.json")
    args = ap.parse_args(argv)

    import tempfile

    from fastapi.testclient import TestClient

    from app.config import settings

    tmp = Path(tempfile.mkdtemp(prefix="blacken-e2e-"))
    settings.upload_dir = tmp / "uploads"
    settings.data_dir = tmp / "data"
    settings.db_path = tmp / "data" / "e2e.db"
    settings.upload_dir.mkdir(parents=True)
    settings.data_dir.mkdir(parents=True)
    # The harness drives hundreds of uploads a minute from one client; the
    # production rate limits would turn most of them into 429s.
    for limit in ("rate_limit_upload_per_min", "rate_limit_redact_per_min", "rate_limit_default_per_min"):
        setattr(settings, limit, 10**9)
    settings.use_calibrated_thresholds = not args.no_calibration

    from app.main import app

    samples = [s for s in ds.load_blacken_gold() if s.meta["domain"] != "negative"]
    if args.ai4privacy:
        samples += ds.load_ai4privacy("test", limit=args.ai4privacy)

    with TestClient(app) as client:
        results = run(client, samples, args.formats, owner_id=f"e2e-{uuid.uuid4()}")

    RESULTS_DIR.mkdir(exist_ok=True)
    suffix = f"_{args.tag}" if args.tag else ""
    out = RESULTS_DIR / f"e2e{suffix}.json"
    out.write_text(json.dumps({"samples": len(samples), "formats": results}, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
