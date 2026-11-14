# Blacken Backend

FastAPI service that detects and redacts personal data (PII) in PDF, Word,
image, and text documents.

## Detection engine

PII detection is built on [Microsoft Presidio](https://microsoft.github.io/presidio/)
(`presidio-analyzer` + `presidio-anonymizer`) - free, MIT-licensed, and runs
entirely locally with no external API calls. It combines:

- A spaCy NLP model (`en_core_web_lg` by default) for PERSON, LOCATION,
  ORGANIZATION, and date/time entities.
- Presidio's built-in, checksum/pattern-validated recognizers for email,
  phone, credit cards, IBAN, crypto wallets, IP addresses, and a wide range
  of country-specific ID formats (US SSN/passport/driver's license, UK NHS,
  India Aadhaar, etc.) - see `app/detection/engine.py`.
- Custom recognizers for domain-specific identifiers not covered by
  Presidio out of the box: medical record numbers, patient/account IDs,
  ages, street addresses, and zip/PIN codes (`app/detection/recognizers.py`).

## Architecture

```
backend/
├── app/
│   ├── config.py            # env-driven settings
│   ├── main.py               # FastAPI app, CORS, startup/cleanup lifespan
│   ├── schemas.py             # request/response models
│   ├── storage.py              # SQLite-backed document store + retention purge
│   ├── utils.py                 # filename sanitization, JSON helpers
│   ├── detection/
│   │   ├── engine.py             # Presidio-based PiiDetectionEngine
│   │   └── recognizers.py         # custom pattern recognizers
│   ├── extraction/
│   │   └── text_extractor.py       # PDF/DOCX/image/txt -> text, OCR fallback
│   ├── redaction/
│   │   ├── text_redactor.py         # bracket redaction for text preview
│   │   ├── pdf_redactor.py           # word/bbox offset-mapped PDF redaction
│   │   ├── docx_redactor.py           # run-level offset-mapped DOCX redaction
│   │   └── image_redactor.py           # OCR line-reconstruction image redaction
│   └── routes/
│       └── documents.py               # all /api/* endpoints
├── run.py                     # dev entrypoint (`python run.py`)
├── requirements.txt
└── tests/                      # pytest suite
```

## Quick Start

```bash
cd backend
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
python -m spacy download en_core_web_lg

python run.py
```

The API is then available at:
- **API**: http://localhost:8000
- **Docs**: http://localhost:8000/api/docs

For production, run `uvicorn app.main:app --host 0.0.0.0 --port 8000`
directly instead of `run.py` (which enables `--reload` for development).

### System dependency: Tesseract OCR

Image and scanned-PDF text extraction need the Tesseract binary (not just
the `pytesseract` pip package):

```bash
brew install tesseract          # macOS
apt-get install tesseract-ocr   # Debian/Ubuntu
```

`GET /api/health` reports whether Tesseract is actually on `PATH`.

## Configuration

Copy `.env.example` to `.env` to override any of these (all have working
defaults):

| Variable | Default | Purpose |
|---|---|---|
| `SPACY_MODEL` | `en_core_web_lg` | NLP model for detection |
| `RETENTION_HOURS` | `6` | how long uploaded documents are kept before auto-deletion |
| `CLEANUP_INTERVAL_MINUTES` | `15` | how often the retention sweep runs |
| `MAX_UPLOAD_MB` | `25` | max upload size |
| `API_KEY` | unset | if set, requires a matching `X-API-Key` header on all `/api/*` routes except `/api/health` |
| `ADMIN_API_KEY` | unset | required as `X-Admin-Key` for `PATCH /api/config`; unset = runtime config changes disabled (they apply to every user) |
| `HOST` / `PORT` | `0.0.0.0` / `8000` | server bind address |
| `CORS_ORIGINS` | localhost only | allowed frontend origins, comma-separated, e.g. `https://your-app.vercel.app,http://localhost:5173` |
| `DATA_ENCRYPTION_KEY` | unset | Fernet key; encrypts stored files and extracted PII at rest. Generate: `python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"` |
| `PII_MODEL_DIR` | unset | directory of the optional GLiNER ONNX model (`scripts/fetch_pii_model.py`) |
| `PII_MODEL_THRESHOLD` | `0.5` | GLiNER span probability threshold |
| `USE_SPACY_NER` | `true` | keep spaCy's own NER alongside GLiNER (recommended - see RESULTS) |
| `MAX_CONCURRENT_JOBS` | `2` | concurrent extract/detect/redact jobs; excess waits, then 503. Use `1` on 512 MB |
| `JOB_QUEUE_TIMEOUT_S` | `30` | how long a request waits for a job slot |
| `RATE_LIMIT_UPLOAD_PER_MIN` / `RATE_LIMIT_REDACT_PER_MIN` / `RATE_LIMIT_DEFAULT_PER_MIN` | `20` / `30` / `240` | per client IP and per owner |
| `OCR_DPI` / `OCR_MIN_CONFIDENCE` | `200` / `0` | scanned-page rasterisation and Tesseract word filter |
| `LOG_FORMAT` | `text` | `json` for log aggregators |
| `SENTRY_DSN` / `ENVIRONMENT` | unset / `development` | error reporting (needs `pip install sentry-sdk`); PII scrubbed |

## Deployment

`Dockerfile` builds a memory-slimmed image, aimed at free-tier hosts like
Railway or Render (typically 512MB-1GB RAM):

- Full `en_core_web_lg` loads to ~705 MB (measured as physical footprint)
  - its 342k-entry word-vector table dominates that. The committed
  `models/en_core_web_lg_pruned` (built once by `scripts/build_pruned_model.py`
  with spaCy's `vocab.prune_vectors(20000)`) loads to ~335 MB.
- Pruning is **not** free, contrary to an earlier note here: on the dev
  splits full `lg` scores about +2 points auto-recall on ai4privacy and +4 on
  Gretel (same precision); on the gold set they are within noise. See
  `evals/RESULTS.md` -> "Model configuration comparison".
- Structured PII - SSN, email, phone, credit card, IP, dates, and this
  repo's custom recognizers - comes from Presidio's pattern/checksum
  recognizers, not spaCy, so model choice only affects PERSON/LOCATION/ORG.

### Choosing a detection profile (all measured, see `evals/RESULTS.md`)

| Profile | Engine memory (peak) | Recall: ai4privacy / Gretel / gold | Gold precision | Detect speed / 10k chars* | Use on |
|---|---:|---|---:|---:|---|
| **Default** - pruned `lg` + rules | 335 MB | 60.0 / 67.9 / 93.7 | 96.5 | ~0.4 s | any host; no model download |
| **Recommended for 512 MB** - `sm` + rules + GLiNER-edge | **215 MB** | **68.4 / 73.7** / 91.1 | 94.8 | ~3.2 s | Railway free tier |
| **Most accurate on the gold set** - pruned `lg` + rules + GLiNER-edge | 431 MB | 65.7 / 69.8 / **95.1** | **96.5** | ~3.2 s | >= 1 GB hosts |
| Full `lg` + rules | 703 MB | 62.3 / 71.9 / 93.5 | 97.3 | ~0.3 s | >= 1.5 GB hosts / HF Space |

Recall columns are 1,000-document dev samples (same sample for every row, so rows compare fairly). On the full held-out test splits the recommended profile scores 65.8 / 74.5 / 91.1 versus 59.3 / 69.1 / 93.7 for the default.

\* measured on Apple Silicon, where onnxruntime int8 is relatively slow; re-measure on the target
host with `python -m evals.run_latency`.

Trade-off of the 512 MB recommendation: the largest recall gains on the 2,000 public
dev documents and the lowest memory, for slightly more false alarms (gold precision
94.8% vs 96.5%; pre-ticked false alarms on 33% vs 25% of PII-free documents). Since a
reviewer unticks a false alarm in one click but can't un-leak a miss, recall wins.

Enable it:

```bash
docker build --build-arg PII_PROFILE=edge -t blacken .
# runtime env
PII_MODEL_DIR=/app/models/gliner-pii-edge
SPACY_MODEL=en_core_web_sm
MAX_CONCURRENT_JOBS=1      # on 512 MB
```

Replacing spaCy's NER with the transformer (`USE_SPACY_NER=false`) was measured and is
**worse** (recall 54-85%): GLiNER-edge complements spaCy rather than replacing it.

## API Endpoints

All `/api/*` routes except `/api/health` require `X-Owner-Id`.

- `GET /api/health`
- `POST /api/upload`
- `POST /api/redact`
- `GET /api/download/{document_id}`
- `GET /api/document/{document_id}/preview`
- `GET /api/document/{document_id}/redacted-preview`
- `GET /api/document/{document_id}/audit-log`
- `GET /api/document/{document_id}/download-audit-log` (entity values masked)
- `GET /api/document/{document_id}/page/{n}?redacted=true` - PNG page preview (PDF/images)
- `DELETE /api/document/{document_id}`
- `GET /api/documents`
- `GET` / `PUT /api/preferences` - per-owner "always redact" / "never flag" terms
- `GET /api/config`, `PATCH /api/config` (admin)
- `GET /api/admin/quality` - real-world precision from reviewer decisions (admin)
- `GET /api/admin/metrics` - Prometheus metrics (admin; `X-Admin-Key` or `Authorization: Bearer`)

## Data handling

- Uploaded files live in `backend/uploads/`; document metadata (extracted
  text, detected entities, redaction state) lives in a local SQLite file at
  `backend/data/blacken.db`.
- A background task purges documents (DB rows + files) older than
  `RETENTION_HOURS` on an interval of `CLEANUP_INTERVAL_MINUTES`. This is a
  PII-handling tool, so documents are not retained indefinitely by default.
- Uploaded filenames are sanitized (no path traversal), and both file
  extension and size are validated on upload.
- Every `/api/*` route except `/api/health` requires an `X-Owner-Id` header
  (>= 16 chars; the frontend sends a per-browser UUID). Documents are scoped
  to it, and requests without one get `400` - there is no shared fallback
  bucket.
- Redacted PDFs and DOCX files have their document metadata (author, title,
  XMP...) scrubbed; DOCX redaction covers tables, headers, footers and
  hyperlinks, not just body paragraphs.

## Testing

```bash
source venv/bin/activate
SPACY_MODEL=models/en_core_web_lg_pruned pytest -v
```

`tests/test_quality_gate.py` fails the build if recall/precision on the
hand-annotated gold set regresses, and `tests/test_production_hardening.py`
pins every defect the evaluation harness found.

## Evaluation

`evals/` measures detection precision/recall/F1/F2 on public benchmarks
(ai4privacy, Gretel) and a hand-annotated gold set, end-to-end leakage in real
redacted PDF/DOCX/PNG files, and latency. See `evals/README.md` for
methodology and `evals/RESULTS.md` for current numbers.

`tests/test_detection.py` covers detection accuracy and false-positive
regressions directly against the engine. `tests/test_redaction_e2e.py`
drives the real HTTP API end to end (upload -> redact -> download) and
verifies the PII is actually gone from the output file, not just flagged.
