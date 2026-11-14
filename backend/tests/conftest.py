"""Shared pytest fixtures. `pythonpath = .` in pytest.ini (backend/) makes
the `app` package importable without a manual sys.path hack.

The detection model comes from settings (SPACY_MODEL), so tests run against
whatever the deployment ships - e.g. `SPACY_MODEL=models/en_core_web_lg_pruned
pytest` in CI, with no 560MB download.
"""

import pytest

from app.config import settings
from app.detection.engine import PiiDetectionEngine

TEST_OWNER_ID = "test-owner-0000000000000001"


@pytest.fixture(scope="session")
def engine() -> PiiDetectionEngine:
    return PiiDetectionEngine(spacy_model=settings.spacy_model)


@pytest.fixture()
def api_client(tmp_path, monkeypatch):
    """A TestClient wired to isolated, per-test upload/data directories so
    tests never touch the real backend/uploads or backend/data. Sends a valid
    X-Owner-Id by default, like the frontend does."""
    monkeypatch.setattr(settings, "upload_dir", tmp_path / "uploads")
    monkeypatch.setattr(settings, "data_dir", tmp_path / "data")
    monkeypatch.setattr(settings, "db_path", tmp_path / "data" / "test.db")
    monkeypatch.setattr(settings, "retention_hours", 1_000_000.0)
    for limit in ("rate_limit_upload_per_min", "rate_limit_redact_per_min", "rate_limit_default_per_min"):
        monkeypatch.setattr(settings, limit, 10_000)
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    settings.data_dir.mkdir(parents=True, exist_ok=True)

    from fastapi.testclient import TestClient

    from app.main import app
    from app.security import rate_limiter

    rate_limiter.reset()

    with TestClient(app, headers={"X-Owner-Id": TEST_OWNER_ID}) as client:
        yield client
