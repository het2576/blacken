"""FastAPI application entrypoint: logging/observability, CORS, lifespan
(load the detector, open the encrypted document store, start the retention
cleanup loop), and route wiring."""

import asyncio
import logging
import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.detection.engine import PiiDetectionEngine
from app.observability import RequestContextMiddleware, configure_logging, init_sentry
from app.routes.documents import api_router
from app.security import JobLimiter
from app.storage import DocumentStore
from app.vault import Vault

configure_logging(settings.log_level, json_logs=settings.log_format == "json")
logger = logging.getLogger("blacken")


def _purge_stray_audit_logs() -> None:
    """Older versions wrote audit-log downloads to disk; sweep any leftovers
    by file age. (Audit logs are now streamed from memory.)"""
    cutoff_seconds = settings.retention_hours * 3600
    now = time.time()
    for path in settings.data_dir.glob("audit_log_*.json"):
        try:
            if now - path.stat().st_mtime > cutoff_seconds:
                path.unlink()
        except OSError:
            continue


async def _purge_expired_documents(store: DocumentStore) -> None:
    expired = await store.purge_expired(settings.retention_hours)
    for doc in expired:
        for path_key in ("original_path", "redacted_path"):
            path = doc.get(path_key)
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                except OSError:
                    logger.warning("Failed to remove expired file for document %s", doc.get("id"))

    _purge_stray_audit_logs()

    if expired:
        logger.info("Retention cleanup: purged %d expired document(s)", len(expired))


async def _cleanup_loop(app: FastAPI) -> None:
    interval_seconds = settings.cleanup_interval_minutes * 60
    while True:
        try:
            await asyncio.sleep(interval_seconds)
            await _purge_expired_documents(app.state.store)
        except asyncio.CancelledError:
            break
        except Exception:  # noqa: BLE001
            logger.exception("Retention cleanup loop iteration failed")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Blacken API...")
    if init_sentry(settings.sentry_dsn or "", settings.environment):
        logger.info("Sentry error reporting enabled (PII scrubbing on)")
    app.state.vault = Vault(settings.data_encryption_key)
    if not app.state.vault.enabled:
        logger.warning("DATA_ENCRYPTION_KEY not set - documents are stored unencrypted at rest")
    app.state.store = DocumentStore(settings.db_path, vault=app.state.vault)
    app.state.jobs = JobLimiter(settings.max_concurrent_jobs, settings.job_queue_timeout_s)
    app.state.engine = PiiDetectionEngine(
        spacy_model=settings.spacy_model,
        language=settings.detection_language,
        pii_model_dir=settings.pii_model_dir,
        pii_model_threshold=settings.pii_model_threshold,
        use_spacy_ner=settings.use_spacy_ner,
        use_calibrated_thresholds=settings.use_calibrated_thresholds,
    )

    # Purge anything already past retention from a previous run before
    # starting the periodic loop.
    await _purge_expired_documents(app.state.store)
    cleanup_task = asyncio.create_task(_cleanup_loop(app))

    logger.info("Blacken API ready.")
    yield

    cleanup_task.cancel()
    try:
        await cleanup_task
    except asyncio.CancelledError:
        pass
    logger.info("Blacken API shut down.")


app = FastAPI(
    title="Blacken API",
    description="AI-powered document PII detection and redaction",
    version="2.1.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    lifespan=lifespan,
)

app.add_middleware(RequestContextMiddleware, json_logs=settings.log_format == "json")
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Request-ID", "X-Page-Count", "Retry-After"],
)

app.include_router(api_router)
