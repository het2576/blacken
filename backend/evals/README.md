# Blacken evaluation harness

Measures what matters for a redaction tool: **how much personal data survives**
— in the detector's output and in the actual redacted files — plus latency.
Results: [`RESULTS.md`](RESULTS.md) (generated, never hand-edited).

```bash
cd backend
pip install -r evals/requirements.txt
export SPACY_MODEL=models/en_core_web_lg_pruned   # what production ships

python -m evals.run_detection                 # detection P/R/F1/F2, all test sets
python -m evals.run_e2e                       # leakage in real PDF/DOCX/PNG output
python -m evals.run_latency                   # latency + memory (add --live-url URL)
python -m evals.report                        # -> evals/RESULTS.md
pytest tests/test_quality_gate.py             # fast CI gate on the gold set

# Model / threshold experiments
python scripts/fetch_pii_model.py             # GLiNER-edge (pinned, SHA-256 checked)
python -m evals.run_detection --split dev --limit 1000 --model en_core_web_sm \
    --pii-model models/gliner-pii-edge --tag cfg_F_sm_edge_ner
python -m evals.calibrate --limit 1500 --dry-run   # per-type thresholds (dev only)
```

Every experiment is judged on the **test** splits *and* the fully annotated gold
set before it ships. Two were rejected this way (see RESULTS.md): dev-fitted
thresholds (gold recall -4.3 pts - they learned the public sets' label gaps) and
300 DPI OCR (no gain, 2.25x memory). Their artifacts live in `results/rejected/`.

## Datasets

| Set | Size | What it tests | Split discipline |
|---|---|---|---|
| **Blacken gold** (`data/blacken_gold.txt`) | 44 docs, 249 PII entities | Realistic resumes, clinical notes, Indian KYC/loan/rent/payslips, US HR/tax/lease, UK NHS/visa, support chats, server logs; 12 PII-free negatives | Test only. Every PII occurrence annotated → precision is meaningful |
| **ai4privacy pii-masking-400k** (EN) | 17,046 docs, 17.7k entities | Breadth of identifier formats, US + GB locales | Tuned on `train` (dev), reported on `validation` (test) |
| **Gretel synthetic PII finance** (EN) | 2,891 docs, 9.1k entities | Long financial/legal documents (~1.4k chars) | Tuned on `train` (dev), reported on `test` |

Public sets are downloaded on first use into `evals/.cache/` (gitignored).

Gold-set markup is inline — `[[PERSON|Asha Rao]]` — so offsets are computed,
never hand-counted. Annotation rules: names, contact details, addresses down to
city/postcode, full dates tied to a person, ages, and every identifier are PII.
Countries/states, years and month-year ranges, company hotlines, invoice/ticket
numbers and timestamps are `IGNORE` (don't-care: neither rewarded nor
penalised). Organizations are scored separately (not personal data).
Identifiers are checksum-valid fakes (Verhoeff Aadhaar, mod-11 NHS, Luhn cards,
mod-97 IBAN), because real documents carry valid checksums and Presidio
validates them.

## Metrics

All datasets map onto one class space (`taxonomy.py`): PERSON, EMAIL, PHONE,
ADDRESS, DATE, GOV_ID, FINANCIAL, OTHER_ID, NETWORK, AGE (+ ORGANIZATION,
IGNORE).

* **Token-level P/R/F1/F2** (headline) — type-agnostic, per alphanumeric
  token; the Presidio-research standard and the closest proxy for leakage.
* **Entity overlap P/R** — is each PII item flagged at all?
* **Full-coverage recall** — every token of the entity covered (catching
  "Smith" but not "John" still leaks).
* **Strict F1** — exact boundaries + class, seqeval-style.
* **F2** weights recall 2× — a leaked SSN costs more than an extra black bar.
* **95% CIs** — document-level percentile bootstrap (errors within a document
  are correlated, so tokens are not resampled independently).
* Two operating points: `detected` (everything the reviewer sees) and `auto`
  (pre-ticked; what one-click Redact removes — the number that matters).

End-to-end (`run_e2e.py`) renders each gold document into PDF, scanned PDF,
DOCX, DOCX-with-tables-and-header, and PNG; drives upload → redact → download
through the real API; and extracts everything recoverable from the output (PDF
text layer + metadata, every DOCX XML part, OCR for images). It reports
`leak_rate_auto`, `leak_rate_oracle` (gold entities fed straight to the
redactor — isolates redaction bugs from detection misses), over-redaction
(non-PII words destroyed) and metadata leaks. For OCR'd formats a value only
counts if it was readable in the *unredacted* rendering.

## Caveats (read before quoting numbers)

* **Public-set precision is a lower bound.** Gretel leaves much real PII
  unlabelled (e.g. "Account Number: 123456789", placeholder phone
  `1234567890` ×583) and ai4privacy doesn't label countries; correct
  detections of those count as false positives. Quote recall from public sets,
  precision from the gold set.
* **ai4privacy values are often structurally invalid** (phone
  "+15-95069-4142", bare building number "493" as an address). Blacken
  validates phones with libphonenumber by design, so it scores lower here than
  a model trained on this distribution would. We deliberately did not tune
  toward it.
* **The gold set was written by the same team that tuned the engine.** Its
  errors informed which *classes* of rule were added (labelled fields,
  country ID recognizers), so treat it as a regression suite more than an
  unbiased estimate; the public test splits were never used for tuning.
* Usernames, passwords, API keys, PINs and geo-coordinates are out of scope
  (`IGNORE`).
