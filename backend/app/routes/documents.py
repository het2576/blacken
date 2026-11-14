"""API routes.

Paths and response shapes of the original endpoints match the frontend
contract in src/services/api.ts. Added since: preferences (custom terms),
page-image previews, reviewer-feedback quality stats and Prometheus metrics
(both admin-only).

Request pipeline for the heavy endpoints:
  owner check -> rate limit -> content validation -> job slot (bounded
  concurrency) -> work in a thread (never on the event loop) -> encrypted
  persistence.
"""

import asyncio
import io
import json
import logging
import math
import mimetypes
import os
import secrets
import shutil
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, Depends, File, Header, HTTPException, Request, UploadFile
from fastapi.responses import PlainTextResponse, Response
from pydantic import BaseModel, Field

from app.config import settings
from app.detection.engine import PiiDetectionEngine
from app.extraction.text_extractor import ExtractionError, extract_text
from app.observability import metrics
from app.redaction.docx_redactor import redact_docx
from app.redaction.image_redactor import redact_image
from app.redaction.matching import find_matches
from app.redaction.pdf_redactor import redact_pdf
from app.redaction.text_redactor import redact_text, redact_txt_file
from app.schemas import ConfigUpdateRequest, RedactionRequest
from app.security import JobLimiter, enforce_rate_limit, validate_file_content
from app.storage import DocumentStore
from app.utils import make_json_serializable, sanitize_filename
from app.vault import Vault

logger = logging.getLogger("blacken")

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tiff", ".bmp"}
REDACTORS = {
    ".pdf": redact_pdf,
    ".docx": redact_docx,
    ".txt": redact_txt_file,
    **{ext: redact_image for ext in IMAGE_EXTENSIONS},
}
MAX_CUSTOM_TERMS = 200
MAX_CUSTOM_TERM_LENGTH = 200


# --------------------------------------------------------------- deps
def get_store(request: Request) -> DocumentStore:
    return request.app.state.store


def get_engine(request: Request) -> PiiDetectionEngine:
    return request.app.state.engine


def get_vault(request: Request) -> Vault:
    return request.app.state.vault


def get_jobs(request: Request) -> JobLimiter:
    return request.app.state.jobs


async def require_api_key(x_api_key: Optional[str] = Header(default=None)) -> None:
    """No-op unless the API_KEY env var is set; then callers must send a
    matching X-API-Key header."""
    if settings.api_key and not secrets.compare_digest(x_api_key or "", settings.api_key):
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


MIN_OWNER_ID_LENGTH = 16


def get_owner_id(x_owner_id: Optional[str] = Header(default=None)) -> str:
    """Every document is scoped to the anonymous owner token the browser sends
    in the X-Owner-Id header (a UUID persisted in localStorage). The app has
    no login, so this per-client token is the isolation boundary.

    A missing or trivially short token is rejected outright. The previous
    version substituted a fixed sentinel for a missing header - but used that
    same sentinel on *upload* too, so every header-less client wrote into, and
    could list, one shared bucket of other people's documents. Short tokens
    are rejected because a guessable owner id ("1", "test") is the same bug."""
    owner = (x_owner_id or "").strip()
    if len(owner) < MIN_OWNER_ID_LENGTH:
        raise HTTPException(
            status_code=400,
            detail=f"X-Owner-Id header is required (at least {MIN_OWNER_ID_LENGTH} characters)",
        )
    return owner


async def require_admin_key(
    x_admin_key: Optional[str] = Header(default=None),
    authorization: Optional[str] = Header(default=None),
) -> None:
    """Admin endpoints (global config, quality stats, metrics) need
    ADMIN_API_KEY configured and presented as X-Admin-Key or as a Bearer
    token (the form Prometheus scrapers send)."""
    if not settings.admin_api_key:
        raise HTTPException(
            status_code=403,
            detail="Admin endpoints are disabled (set ADMIN_API_KEY to enable)",
        )
    presented = x_admin_key or ""
    if not presented and authorization and authorization.lower().startswith("bearer "):
        presented = authorization[7:]
    if not secrets.compare_digest(presented, settings.admin_api_key):
        raise HTTPException(status_code=401, detail="Invalid or missing admin key")


def rate_limited(bucket: str, limit_attr: str):
    def dependency(request: Request) -> None:
        enforce_rate_limit(request, bucket, getattr(settings, limit_attr))

    return dependency


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


public_router = APIRouter()
router = APIRouter(
    dependencies=[Depends(require_api_key), Depends(rate_limited("api", "rate_limit_default_per_min"))]
)
admin_router = APIRouter(prefix="/admin", dependencies=[Depends(require_admin_key)])


# --------------------------------------------------------------- helpers
def _apply_custom_terms(text: str, entities: List[dict], prefs: dict) -> List[dict]:
    """Per-owner overrides on top of the detector:
    * never_flag - drop detections whose text matches (e.g. the user's own
      company name, flagged on every page of every document);
    * always_redact - add every boundary-respecting occurrence of the term as
      a pre-selected CUSTOM_TERM entity (e.g. a client's name the model
      doesn't recognise), replacing any overlapping detection."""
    never = {t.casefold() for t in prefs.get("never_flag", [])}
    if never:
        entities = [e for e in entities if e["text"].strip().casefold() not in never]
    custom = []
    for term in prefs.get("always_redact", []):
        for start, end in find_matches(text, [term]):
            custom.append(
                {
                    "text": text[start:end],
                    "type": "CUSTOM_TERM",
                    "start": start,
                    "end": end,
                    "confidence": 1.0,
                    "source": "custom_term",
                    "selected": True,
                }
            )
    if not custom:
        return entities
    kept = [e for e in entities if not any(e["start"] < c["end"] and c["start"] < e["end"] for c in custom)]
    return sorted(kept + custom, key=lambda e: e["start"])


def _feedback_events(detected: List[dict], submitted: List[dict]) -> List[dict]:
    """Turn a reviewer's submitted selection into PII-free feedback rows.
    accepted = detected and left ticked; rejected = detected but unticked;
    added = submitted but never detected (a miss the reviewer caught)."""
    by_key = {(e["start"], e["end"], e["type"]): e for e in submitted}
    events = []
    for e in detected:
        key = (e["start"], e["end"], e["type"])
        sub = by_key.pop(key, None)
        outcome = "accepted" if (sub is not None and sub.get("selected", True)) else "rejected"
        events.append(
            {
                "entity_type": e["type"],
                "confidence_bucket": math.floor(float(e.get("confidence", 0)) * 10) / 10,
                "outcome": outcome,
                "was_preselected": int(bool(e.get("selected"))),
            }
        )
    for sub in by_key.values():
        if sub.get("selected", True):
            events.append({"entity_type": sub["type"], "confidence_bucket": None, "outcome": "added", "was_preselected": 0})
    return events


def _masked_entities(entities: List[dict]) -> List[dict]:
    """Audit records prove *what kind* of data was found and where, without
    re-exposing the values themselves (the old audit log stored every SSN and
    name in clear, defeating the redaction it was auditing)."""
    return [
        {
            "type": e.get("type"),
            "start": e.get("start"),
            "end": e.get("end"),
            "length": len(e.get("text") or ""),
            "confidence": e.get("confidence"),
            "selected": e.get("selected", True),
            "source": e.get("source"),
        }
        for e in entities
    ]


async def _get_doc_or_404(store: DocumentStore, document_id: str, owner_id: str) -> dict:
    doc = await store.get(document_id, owner_id)
    if doc is None:
        raise HTTPException(status_code=404, detail="Document not found")
    return doc


def _remove_files(doc: dict) -> None:
    for path_key in ("original_path", "redacted_path"):
        path = doc.get(path_key)
        if path and os.path.exists(path):
            try:
                os.remove(path)
            except OSError:
                logger.warning("Failed to remove file for document %s", doc.get("id"))


# --------------------------------------------------------------- health
@public_router.get("/health")
async def health_check(request: Request):
    engine = getattr(request.app.state, "engine", None)
    return {
        "status": "healthy" if engine is not None else "degraded",
        "timestamp": _now_iso(),
        "features": {
            "ocr": shutil.which("tesseract") is not None,
            "pdf": True,
            "docx": True,
            "detection_engine": engine is not None,
            "transformer_model": bool(getattr(engine, "pii_model_enabled", False)),
            "encryption_at_rest": request.app.state.vault.enabled,
        },
    }


# --------------------------------------------------------------- documents
@router.get("/documents")
async def list_documents(
    store: DocumentStore = Depends(get_store),
    owner_id: str = Depends(get_owner_id),
):
    docs = await store.list_all(owner_id)
    retention_seconds = settings.retention_hours * 3600
    documents = [
        {
            "document_id": doc["id"],
            "filename": doc["filename"],
            "upload_time": doc["upload_time"],
            "status": doc.get("status", "processed"),
            "entity_count": len(doc.get("entities", [])),
            "redacted_count": doc.get("redacted_count") or 0,
            "expires_at": datetime.fromtimestamp(doc["created_at"] + retention_seconds, tz=timezone.utc)
            .isoformat()
            .replace("+00:00", "Z"),
        }
        for doc in docs
    ]
    return {"success": True, "documents": documents}


@router.post("/upload", dependencies=[Depends(rate_limited("upload", "rate_limit_upload_per_min"))])
async def upload_document(
    file: UploadFile = File(...),
    store: DocumentStore = Depends(get_store),
    engine: PiiDetectionEngine = Depends(get_engine),
    vault: Vault = Depends(get_vault),
    jobs: JobLimiter = Depends(get_jobs),
    owner_id: str = Depends(get_owner_id),
):
    safe_name = sanitize_filename(file.filename or "upload")
    ext = Path(safe_name).suffix.lower()
    if ext not in settings.allowed_extensions:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext or 'unknown'}")

    max_bytes = int(settings.max_upload_mb * 1024 * 1024)
    content = await file.read(max_bytes + 1)  # never buffer more than the limit
    if len(content) > max_bytes:
        raise HTTPException(status_code=413, detail=f"File too large (max {settings.max_upload_mb} MB)")
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")
    validate_file_content(ext, content)

    document_id = str(uuid.uuid4())
    upload_path = settings.upload_dir / f"{document_id}{ext}"
    await asyncio.to_thread(vault.write_file, upload_path, content)
    del content

    try:
        async with jobs.slot():
            with vault.plaintext_path(str(upload_path), suffix=ext) as plain:
                extracted_text = await extract_text(plain, safe_name)
            # Detection is CPU-bound; inline in this async handler it would
            # freeze the event loop for every other request.
            entities = await asyncio.to_thread(engine.detect, extracted_text)
    except ExtractionError as exc:
        upload_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except BaseException:
        upload_path.unlink(missing_ok=True)
        raise

    prefs = await store.get_prefs(owner_id)
    entities = make_json_serializable(_apply_custom_terms(extracted_text, entities, prefs))

    await store.create(
        document_id,
        {
            "owner_id": owner_id,
            "filename": safe_name,
            "upload_time": _now_iso(),
            "original_path": str(upload_path),
            "extracted_text": extracted_text,
            "entities": entities,
            "status": "processed",
        },
    )

    metrics.inc("blacken_documents_processed_total", format=ext.lstrip("."))
    for e in entities:
        metrics.inc("blacken_entities_detected_total", type=e["type"])
    logger.info("Uploaded document %s (%s) -> %d entities", document_id, ext, len(entities))

    return {
        "success": True,
        "document_id": document_id,
        "filename": safe_name,
        "extracted_text": extracted_text,
        "entities": entities,
        "entity_count": len(entities),
    }


@router.post("/redact", dependencies=[Depends(rate_limited("redact", "rate_limit_redact_per_min"))])
async def redact_document(
    request: RedactionRequest,
    store: DocumentStore = Depends(get_store),
    vault: Vault = Depends(get_vault),
    jobs: JobLimiter = Depends(get_jobs),
    owner_id: str = Depends(get_owner_id),
):
    doc_info = await _get_doc_or_404(store, request.document_id, owner_id)
    original_text = doc_info["extracted_text"]

    if not request.entities:
        entities_to_redact = doc_info.get("entities", [])
    else:
        submitted = [e.model_dump() for e in request.entities]
        await store.record_feedback(_feedback_events(doc_info.get("entities", []), submitted))
        entities_to_redact = [
            {"text": e["text"], "type": e["type"], "start": e["start"], "end": e["end"], "selected": True}
            for e in submitted
            if e["selected"]
        ]

    selected_count = sum(1 for e in entities_to_redact if e.get("selected", True))
    redacted_text = redact_text(original_text, entities_to_redact)

    original_filename = doc_info["filename"]
    file_ext = Path(original_filename).suffix.lower()
    redactor = REDACTORS.get(file_ext)
    if redactor is None:
        raise HTTPException(status_code=400, detail=f"File type {file_ext} not supported for redaction")

    fd, tmp_name = tempfile.mkstemp(suffix=file_ext, prefix=f"{request.document_id}_", dir=settings.data_dir)
    os.close(fd)
    redacted_path = Path(tmp_name)
    try:
        async with jobs.slot():
            with vault.plaintext_path(doc_info["original_path"], suffix=file_ext) as plain:
                # CPU-bound (rendering, OCR) - keep it off the event loop.
                await asyncio.to_thread(redactor, plain, entities_to_redact, str(redacted_path))
        if not redacted_path.exists() or redacted_path.stat().st_size == 0:
            raise HTTPException(status_code=500, detail="Redaction produced no output file")
        await asyncio.to_thread(vault.encrypt_file_in_place, redacted_path)
    except HTTPException:
        redacted_path.unlink(missing_ok=True)
        raise
    except Exception as exc:  # noqa: BLE001
        redacted_path.unlink(missing_ok=True)
        logger.exception("Redaction failed for document %s", request.document_id)
        raise HTTPException(status_code=500, detail="Redaction failed") from exc

    # Replacing a previous redaction: don't leave the old output behind.
    old = doc_info.get("redacted_path")
    if old and old != str(redacted_path) and os.path.exists(old):
        os.remove(old)

    await store.update(
        request.document_id,
        redacted_text=redacted_text,
        redacted_path=str(redacted_path),
        redacted_filename=f"redacted_{Path(original_filename).stem}{file_ext}",
        redacted_count=selected_count,
        redacted_entities=entities_to_redact,
        redaction_time=_now_iso(),
        status="redacted",
    )
    metrics.inc("blacken_documents_redacted_total", format=file_ext.lstrip("."))

    return {
        "success": True,
        "document_id": request.document_id,
        "redacted_text": redacted_text,
        "redacted_count": selected_count,
        "download_url": f"/api/download/{request.document_id}",
    }


@router.get("/download/{document_id}")
async def download_document(
    document_id: str,
    original: bool = False,
    store: DocumentStore = Depends(get_store),
    vault: Vault = Depends(get_vault),
    owner_id: str = Depends(get_owner_id),
):
    doc_info = await _get_doc_or_404(store, document_id, owner_id)
    if original:
        file_path, filename = doc_info.get("original_path"), doc_info.get("filename")
        if not file_path or not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="Original file not found")
    else:
        file_path, filename = doc_info.get("redacted_path"), doc_info.get("redacted_filename")
        if not file_path or not os.path.exists(file_path):
            raise HTTPException(
                status_code=404,
                detail="No redacted file available. Document must be successfully redacted before download.",
            )
    data = await asyncio.to_thread(vault.read_file, Path(file_path))
    media_type = mimetypes.guess_type(filename or "")[0] or "application/octet-stream"
    return Response(
        content=data,
        media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="{filename}"', "Cache-Control": "no-store"},
    )


@router.get("/document/{document_id}/page/{page}")
async def render_page_preview(
    document_id: str,
    page: int,
    redacted: bool = True,
    store: DocumentStore = Depends(get_store),
    vault: Vault = Depends(get_vault),
    jobs: JobLimiter = Depends(get_jobs),
    owner_id: str = Depends(get_owner_id),
):
    """PNG of one page of the original or redacted file, so the reviewer sees
    the actual black boxes before downloading. PDFs and images only (a DOCX
    has no fixed page layout without a full Word renderer)."""
    doc_info = await _get_doc_or_404(store, document_id, owner_id)
    path = doc_info.get("redacted_path" if redacted else "original_path")
    if not path or not os.path.exists(path):
        raise HTTPException(status_code=404, detail="File not available - redact the document first")
    ext = Path(doc_info["filename"]).suffix.lower()
    if ext != ".pdf" and ext not in IMAGE_EXTENSIONS:
        raise HTTPException(status_code=415, detail="Page previews are available for PDF and image files")

    def render() -> tuple[bytes, int]:
        data = vault.read_file(Path(path))
        if ext == ".pdf":
            import pymupdf

            with pymupdf.open(stream=data, filetype="pdf") as pdf:
                if not 1 <= page <= pdf.page_count:
                    raise HTTPException(status_code=404, detail="Page out of range")
                return pdf[page - 1].get_pixmap(dpi=100).tobytes("png"), pdf.page_count
        from PIL import Image

        if page != 1:
            raise HTTPException(status_code=404, detail="Page out of range")
        img = Image.open(io.BytesIO(data)).convert("RGB")
        img.thumbnail((1400, 1400))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue(), 1

    async with jobs.slot():
        png, page_count = await asyncio.to_thread(render)
    return Response(
        content=png,
        media_type="image/png",
        headers={"X-Page-Count": str(page_count), "Cache-Control": "no-store"},
    )


@router.delete("/document/{document_id}")
async def delete_document(
    document_id: str,
    store: DocumentStore = Depends(get_store),
    owner_id: str = Depends(get_owner_id),
):
    doc_info = await _get_doc_or_404(store, document_id, owner_id)
    _remove_files(doc_info)
    await store.delete(document_id, owner_id)
    logger.info("Deleted document %s on request", document_id)
    return {"success": True, "document_id": document_id}


@router.get("/document/{document_id}/preview")
async def get_document_preview(
    document_id: str,
    store: DocumentStore = Depends(get_store),
    owner_id: str = Depends(get_owner_id),
):
    doc_info = await _get_doc_or_404(store, document_id, owner_id)
    extracted_text = doc_info.get("extracted_text") or ""
    text_preview = extracted_text[:500] + "..." if len(extracted_text) > 500 else extracted_text
    return {
        "success": True,
        "document_id": document_id,
        "filename": doc_info["filename"],
        "upload_time": doc_info["upload_time"],
        "status": doc_info.get("status", "processed"),
        "entity_count": len(doc_info.get("entities", [])),
        "extracted_text_preview": text_preview,
        "entities": doc_info.get("entities", []),
        "redacted_count": doc_info.get("redacted_count") or 0,
    }


@router.get("/document/{document_id}/redacted-preview")
async def get_redacted_document_preview(
    document_id: str,
    store: DocumentStore = Depends(get_store),
    owner_id: str = Depends(get_owner_id),
):
    doc_info = await _get_doc_or_404(store, document_id, owner_id)
    if doc_info.get("status") != "redacted":
        raise HTTPException(status_code=400, detail="Document has not been redacted yet")
    redacted_text = doc_info.get("redacted_text") or ""
    return {
        "success": True,
        "document_id": document_id,
        "filename": doc_info["filename"],
        "redacted_filename": doc_info.get("redacted_filename") or f"redacted_{doc_info['filename']}",
        "redacted_count": doc_info.get("redacted_count") or 0,
        "redacted_text_preview": redacted_text[:500] + "..." if len(redacted_text) > 500 else redacted_text,
        "download_url": f"/api/download/{document_id}",
        "status": doc_info.get("status", "redacted"),
    }


def _build_audit_entries(doc_info: dict) -> list[dict]:
    entries = [
        {
            "timestamp": doc_info["upload_time"],
            "action": "DOCUMENT_UPLOADED",
            "details": f"Uploaded file: {doc_info['filename']}",
            "status": "completed",
        }
    ]
    entities = doc_info.get("entities", [])
    if entities:
        entries.append(
            {
                "timestamp": doc_info["upload_time"],
                "action": "ENTITIES_DETECTED",
                "details": f"Detected {len(entities)} sensitive entities",
                "status": "completed",
                "entities": _masked_entities(entities),
            }
        )
    if doc_info.get("status") == "redacted":
        entries.append(
            {
                "timestamp": doc_info.get("redaction_time") or doc_info["upload_time"],
                "action": "DOCUMENT_REDACTED",
                "details": f"Redacted {doc_info.get('redacted_count') or 0} entities",
                "status": "completed",
                "redacted_count": doc_info.get("redacted_count") or 0,
                "redacted_entities": _masked_entities(doc_info.get("redacted_entities", [])),
            }
        )
    return entries


@router.get("/document/{document_id}/audit-log")
async def get_audit_log(
    document_id: str,
    store: DocumentStore = Depends(get_store),
    owner_id: str = Depends(get_owner_id),
):
    doc_info = await _get_doc_or_404(store, document_id, owner_id)
    audit_entries = _build_audit_entries(doc_info)
    return {
        "success": True,
        "document_id": document_id,
        "filename": doc_info["filename"],
        "audit_entries": audit_entries,
        "total_entries": len(audit_entries),
    }


@router.get("/document/{document_id}/download-audit-log")
async def download_audit_log(
    document_id: str,
    store: DocumentStore = Depends(get_store),
    owner_id: str = Depends(get_owner_id),
):
    """Streamed straight from memory - the old version wrote each audit log
    to disk first, leaving extra copies of document metadata lying around."""
    doc_info = await _get_doc_or_404(store, document_id, owner_id)
    redacted_text = doc_info.get("redacted_text") or ""
    audit_log = {
        "document_id": document_id,
        "filename": doc_info["filename"],
        "upload_time": doc_info["upload_time"],
        "status": doc_info.get("status", "processed"),
        "audit_timestamp": _now_iso(),
        "entities_detected": len(doc_info.get("entities", [])),
        "entities_redacted": doc_info.get("redacted_count") or 0,
        "entities": _masked_entities(doc_info.get("entities", [])),
        "redacted_text_preview": redacted_text[:1000] + "..." if len(redacted_text) > 1000 else redacted_text,
        "audit_entries": _build_audit_entries(doc_info),
    }
    filename = f"audit_log_{document_id}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
    return Response(
        content=json.dumps(audit_log, indent=2, ensure_ascii=False),
        media_type="application/json",
        headers={"Content-Disposition": f"attachment; filename={filename}", "Cache-Control": "no-store"},
    )


# --------------------------------------------------------------- preferences
class Preferences(BaseModel):
    always_redact: List[str] = Field(default_factory=list)
    never_flag: List[str] = Field(default_factory=list)


def _clean_terms(terms: List[str]) -> List[str]:
    out, seen = [], set()
    for t in terms:
        t = " ".join((t or "").split())
        if len(t) < 2 or len(t) > MAX_CUSTOM_TERM_LENGTH or t.casefold() in seen:
            continue
        seen.add(t.casefold())
        out.append(t)
    if len(out) > MAX_CUSTOM_TERMS:
        raise HTTPException(status_code=400, detail=f"At most {MAX_CUSTOM_TERMS} terms per list")
    return out


@router.get("/preferences")
async def get_preferences(store: DocumentStore = Depends(get_store), owner_id: str = Depends(get_owner_id)):
    return await store.get_prefs(owner_id)


@router.put("/preferences")
async def put_preferences(
    prefs: Preferences,
    store: DocumentStore = Depends(get_store),
    owner_id: str = Depends(get_owner_id),
):
    always, never = _clean_terms(prefs.always_redact), _clean_terms(prefs.never_flag)
    await store.set_prefs(owner_id, always, never)
    return {"always_redact": always, "never_flag": never}


# --------------------------------------------------------------- config
def _config_snapshot() -> dict:
    return {
        "retention_hours": settings.retention_hours,
        "cleanup_interval_minutes": settings.cleanup_interval_minutes,
        "max_upload_mb": settings.max_upload_mb,
        "allowed_extensions": sorted(settings.allowed_extensions),
        "detection_language": settings.detection_language,
        "spacy_model": Path(settings.spacy_model).name,
        "api_key_required": settings.api_key is not None,
        "config_editable": settings.admin_api_key is not None,
    }


@router.get("/config")
async def get_config():
    return _config_snapshot()


@router.patch("/config", dependencies=[Depends(require_admin_key)])
async def update_config(update: ConfigUpdateRequest):
    if update.retention_hours is not None:
        if not (0.1 <= update.retention_hours <= 720):
            raise HTTPException(status_code=400, detail="retention_hours must be between 0.1 and 720")
        settings.retention_hours = update.retention_hours
    if update.max_upload_mb is not None:
        if not (0.1 <= update.max_upload_mb <= 100):
            raise HTTPException(status_code=400, detail="max_upload_mb must be between 0.1 and 100")
        settings.max_upload_mb = update.max_upload_mb
    logger.info("Runtime config updated: %s", update.model_dump(exclude_none=True))
    return _config_snapshot()


# --------------------------------------------------------------- admin
@admin_router.get("/quality")
async def quality_report(days: float = 30, store: DocumentStore = Depends(get_store)):
    """Real-world precision from reviewer decisions, per entity type and per
    confidence bucket. `precision_of_preselected` is the share of pre-ticked
    entities reviewers kept; `added` counts misses reviewers had to add."""
    rows = await store.feedback_summary(days * 86400 if days else None)
    by_type: dict = {}
    for r in rows:
        t = by_type.setdefault(r["entity_type"], {"accepted": 0, "rejected": 0, "added": 0, "pre_kept": 0, "pre_total": 0, "by_confidence": {}})
        t[r["outcome"]] += r["n"]
        if r["was_preselected"]:
            t["pre_total"] += r["n"]
            t["pre_kept"] += r["n"] if r["outcome"] == "accepted" else 0
        if r["confidence_bucket"] is not None:
            b = t["by_confidence"].setdefault(f"{r['confidence_bucket']:.1f}", {"accepted": 0, "rejected": 0})
            if r["outcome"] in b:
                b[r["outcome"]] += r["n"]
    report = {}
    for etype, t in sorted(by_type.items()):
        reviewed = t["accepted"] + t["rejected"]
        report[etype] = {
            "accepted": t["accepted"],
            "rejected": t["rejected"],
            "added_by_reviewer": t["added"],
            "acceptance_rate": t["accepted"] / reviewed if reviewed else None,
            "precision_of_preselected": t["pre_kept"] / t["pre_total"] if t["pre_total"] else None,
            "by_confidence": t["by_confidence"],
        }
    return {"window_days": days, "entity_types": report}


@admin_router.get("/metrics", response_class=PlainTextResponse)
async def prometheus_metrics(request: Request):
    jobs: JobLimiter = request.app.state.jobs
    return metrics.render({"blacken_jobs_active": jobs.active, "blacken_jobs_waiting": jobs.waiting})


api_router = APIRouter(prefix="/api")
api_router.include_router(public_router)
api_router.include_router(router)
api_router.include_router(admin_router)
