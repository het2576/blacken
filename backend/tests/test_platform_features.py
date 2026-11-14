"""Tests for the production platform features: content validation, rate
limiting, bounded concurrency, encryption at rest, custom terms, reviewer
feedback telemetry, masked audit logs, page previews and observability."""

import asyncio
import io
import sqlite3

import pymupdf
import pytest
from cryptography.fernet import Fernet
from fastapi import HTTPException

from app.config import settings
from app.security import JobLimiter, SlidingWindowRateLimiter

PII = "Patient Name: Robert Johnson\nSSN: 234-56-7890\nEmail: robert.johnson@example.com\nRoutine checkup."


def _pdf(text: str = PII) -> bytes:
    doc = pymupdf.open()
    page = doc.new_page()
    for i, line in enumerate(text.split("\n")):
        page.insert_text((50, 72 + 14 * i), line, fontsize=11)
    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


def _upload(client, name="a.pdf", data=None, **kw):
    return client.post("/api/upload", files={"file": (name, data if data is not None else _pdf(), "application/pdf")}, **kw)


# ------------------------------------------------------------ validation
def test_renamed_binary_is_rejected_by_magic_bytes(api_client):
    r = _upload(api_client, "invoice.pdf", b"MZ\x90\x00 this is really an exe")
    assert r.status_code == 400 and "does not match" in r.json()["detail"]


def test_zip_pretending_to_be_docx_is_rejected(api_client):
    import zipfile

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("hello.txt", "not a word document")
    assert _upload(api_client, "x.docx", buf.getvalue()).status_code == 400


def test_legacy_doc_is_no_longer_accepted(api_client):
    assert _upload(api_client, "old.doc", b"\xd0\xcf\x11\xe0 legacy").status_code == 400


# ------------------------------------------------------------ rate limit / jobs
def test_sliding_window_rate_limiter():
    rl = SlidingWindowRateLimiter()
    assert all(rl.check("k", 3, 60) is None for _ in range(3))
    assert rl.check("k", 3, 60) > 0
    assert rl.check("other", 3, 60) is None


def test_upload_rate_limit_returns_429(api_client, monkeypatch):
    monkeypatch.setattr(settings, "rate_limit_upload_per_min", 2)
    assert _upload(api_client).status_code == 200
    assert _upload(api_client).status_code == 200
    r = _upload(api_client)
    assert r.status_code == 429 and int(r.headers["Retry-After"]) >= 1


def test_job_limiter_sheds_load_with_503():
    async def scenario():
        jobs = JobLimiter(max_concurrent=1, queue_timeout_s=0.05)
        async with jobs.slot():
            with pytest.raises(HTTPException) as exc:
                async with jobs.slot():
                    pass
        assert exc.value.status_code == 503
        async with jobs.slot():  # released correctly
            assert jobs.active == 1

    asyncio.run(scenario())


# ------------------------------------------------------------ encryption
def test_encryption_at_rest_files_and_database(api_client, monkeypatch):
    from app.main import app
    from app.storage import DocumentStore
    from app.vault import Vault

    vault = Vault(Fernet.generate_key().decode())
    app.state.vault = vault
    app.state.store = DocumentStore(settings.db_path, vault=vault)

    up = _upload(api_client)
    assert up.status_code == 200, up.text
    doc_id = up.json()["document_id"]
    assert api_client.post("/api/redact", json={"document_id": doc_id}).status_code == 200

    # Nothing readable on disk...
    for f in settings.upload_dir.iterdir():
        assert not f.read_bytes().startswith(b"%PDF"), f
    raw = sqlite3.connect(settings.db_path).execute(
        "SELECT extracted_text, entities, redacted_text FROM documents WHERE id = ?", (doc_id,)
    ).fetchone()
    for column in raw:
        assert "Robert Johnson" not in column and "234-56-7890" not in column

    # ...but the API still serves correct, decrypted content.
    out = api_client.get(f"/api/download/{doc_id}").content
    assert out.startswith(b"%PDF")
    text = "".join(p.get_text() for p in pymupdf.open(stream=out, filetype="pdf"))
    assert "234-56-7890" not in text and "Routine checkup" in text
    assert api_client.get("/api/health").json()["features"]["encryption_at_rest"] is True


# ------------------------------------------------------------ custom terms
def test_custom_terms_always_redact_and_never_flag(api_client):
    r = api_client.put(
        "/api/preferences",
        json={"always_redact": ["Project Nightjar"], "never_flag": ["Robert Johnson", "x"]},
    )
    assert r.status_code == 200
    assert r.json()["never_flag"] == ["Robert Johnson"]  # too-short term dropped

    up = _upload(api_client, data=_pdf(PII + "\nCodename Project Nightjar is confidential."))
    by_text = {e["text"]: e for e in up.json()["entities"]}
    assert by_text["Project Nightjar"]["type"] == "CUSTOM_TERM" and by_text["Project Nightjar"]["selected"]
    assert "Robert Johnson" not in by_text
    assert "234-56-7890" in by_text

    other = {"X-Owner-Id": "another-owner-0000000000"}
    assert api_client.get("/api/preferences", headers=other).json() == {"always_redact": [], "never_flag": []}


# ------------------------------------------------------------ feedback
def test_reviewer_feedback_feeds_quality_report_without_pii(api_client, monkeypatch):
    monkeypatch.setattr(settings, "admin_api_key", "admin-secret-key")
    up = _upload(api_client).json()
    entities = up["entities"]
    for e in entities:
        e["selected"] = e["type"] != "EMAIL_ADDRESS"  # reviewer unticks the email
    entities.append({"text": "checkup", "type": "PERSON", "start": 0, "end": 7, "selected": True})
    assert api_client.post("/api/redact", json={"document_id": up["document_id"], "entities": entities}).status_code == 200

    assert api_client.get("/api/admin/quality").status_code == 401
    report = api_client.get("/api/admin/quality", headers={"X-Admin-Key": "admin-secret-key"}).json()
    types = report["entity_types"]
    assert types["EMAIL_ADDRESS"]["rejected"] == 1
    assert types["US_SSN"]["accepted"] == 1
    assert types["PERSON"]["added_by_reviewer"] == 1

    rows = sqlite3.connect(settings.db_path).execute("SELECT * FROM feedback").fetchall()
    assert rows and not any("robert" in str(r).lower() for r in rows)


# ------------------------------------------------------------ audit / preview / obs
def test_audit_log_never_contains_raw_pii(api_client):
    doc_id = _upload(api_client).json()["document_id"]
    api_client.post("/api/redact", json={"document_id": doc_id})
    body = api_client.get(f"/api/document/{doc_id}/download-audit-log").text
    assert "234-56-7890" not in body and "Robert Johnson" not in body
    assert '"type": "US_SSN"' in body


def test_page_preview_renders_redacted_png(api_client):
    doc_id = _upload(api_client).json()["document_id"]
    assert api_client.get(f"/api/document/{doc_id}/page/1").status_code == 404  # not redacted yet
    api_client.post("/api/redact", json={"document_id": doc_id})
    r = api_client.get(f"/api/document/{doc_id}/page/1")
    assert r.status_code == 200 and r.content.startswith(b"\x89PNG")
    assert r.headers["X-Page-Count"] == "1"
    assert api_client.get(f"/api/document/{doc_id}/page/2").status_code == 404


def test_request_id_and_metrics(api_client, monkeypatch):
    monkeypatch.setattr(settings, "admin_api_key", "admin-secret-key")
    r = api_client.get("/api/health", headers={"X-Request-ID": "trace-123"})
    assert r.headers["X-Request-ID"] == "trace-123"
    _upload(api_client)
    m = api_client.get("/api/admin/metrics", headers={"Authorization": "Bearer admin-secret-key"})
    assert m.status_code == 200
    assert 'blacken_requests_total{method="POST",route="/api/upload",status="200"}' in m.text
    assert "blacken_documents_processed_total" in m.text
    assert "robert" not in m.text.lower()
    # Concrete ids never appear - only route templates.
    api_client.get("/api/download/some-document-id")
    m = api_client.get("/api/admin/metrics", headers={"X-Admin-Key": "admin-secret-key"}).text
    assert 'route="/api/download/{document_id}"' in m and "some-document-id" not in m
