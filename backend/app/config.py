"""Application configuration, driven by environment variables."""

import json
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # NLP / detection
    spacy_model: str = "en_core_web_lg"
    detection_language: str = "en"
    # Optional GLiNER PII transformer (ONNX, no PyTorch) run alongside spaCy
    # and the pattern recognizers - see app/detection/gliner_onnx.py. Unset =
    # disabled. Point at a dir with model_quint8.onnx + tokenizer.json +
    # gliner_config.json (scripts/fetch_pii_model.py downloads one).
    pii_model_dir: str | None = None
    pii_model_threshold: float = 0.5
    # With the transformer enabled, spaCy's own NER can be switched off and a
    # small spaCy pipeline (en_core_web_sm) used just for tokens/lemmas.
    use_spacy_ner: bool = True
    # Per-type auto-select thresholds from app/detection/thresholds.json
    # (fitted by evals/calibrate.py). False = single global threshold.
    use_calibrated_thresholds: bool = True

    # Storage / retention
    upload_dir: Path = BACKEND_DIR / "uploads"
    data_dir: Path = BACKEND_DIR / "data"
    db_path: Path = BACKEND_DIR / "data" / "blacken.db"
    retention_hours: float = 6.0
    cleanup_interval_minutes: float = 15.0

    # Upload limits
    max_upload_mb: float = 25.0
    # .doc (legacy binary Word) is deliberately absent: python-docx cannot
    # parse it, so it was accepted and then always failed.
    allowed_extensions: frozenset[str] = frozenset(
        {".pdf", ".docx", ".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".txt"}
    )

    # Encryption at rest (Fernet key) for stored files + extracted PII.
    data_encryption_key: str | None = None

    # Abuse protection (requests per minute, per client IP and per owner).
    rate_limit_upload_per_min: int = 20
    rate_limit_redact_per_min: int = 30
    rate_limit_default_per_min: int = 240
    # Concurrent CPU-heavy jobs (extract/detect/redact). 1-2 on a 512 MB box.
    max_concurrent_jobs: int = 2
    job_queue_timeout_s: float = 30.0

    # OCR (scanned PDFs and images). 200 DPI measured equal-or-better than
    # 300 on the e2e suite (evals/RESULTS.md, "OCR") at 2.25x less memory
    # per page raster - the right default for a 512 MB container. Raise it
    # for small-print scans on a bigger host.
    ocr_dpi: int = 200
    ocr_min_confidence: int = 0

    # Observability.
    log_format: str = "text"  # "json" for log aggregators
    sentry_dsn: str | None = None
    environment: str = "development"

    # Security
    api_key: str | None = None
    # Required (as X-Admin-Key) for PATCH /api/config. Unset = runtime config
    # changes disabled, since they apply globally to every user.
    admin_api_key: str | None = None

    # CORS — override via the CORS_ORIGINS env var in production. Plain,
    # comma-separated values are accepted (e.g.
    # `CORS_ORIGINS=https://your-app.vercel.app,http://localhost:5173`) - a
    # JSON array string also still works, but isn't required. Deliberately
    # typed as `str`, not `list[str]`: pydantic-settings auto-JSON-decodes
    # any complex-typed field read from an env var, so a plain
    # comma-separated value in a `list[str]` field crashes the app on
    # startup with a SettingsError before it can even bind to a port -
    # this is parsed by hand in `cors_origins_list` below instead.
    cors_origins: str = (
        "http://localhost:3000,http://localhost:5173,http://localhost:5174,"
        "http://localhost:5175,http://localhost:5176"
    )

    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "INFO"

    @property
    def cors_origins_list(self) -> list[str]:
        raw = self.cors_origins.strip()
        if not raw:
            return []
        if raw.startswith("["):
            try:
                return [str(origin).strip() for origin in json.loads(raw)]
            except (json.JSONDecodeError, TypeError):
                pass
        return [origin.strip() for origin in raw.split(",") if origin.strip()]


settings = Settings()
settings.upload_dir.mkdir(parents=True, exist_ok=True)
settings.data_dir.mkdir(parents=True, exist_ok=True)
