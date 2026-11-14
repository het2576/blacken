"""Tests for the optional transformer and threshold plumbing. The decoding
tests use synthetic logits, so they run in CI without downloading a model;
the integration test runs only when PII_MODEL_DIR points at a real model."""

import json
import os

import numpy as np
import pytest

from app.detection import engine as engine_mod
from app.detection.gliner_onnx import GlinerOnnxRecognizer


def _bare_recognizer(labels, threshold=0.5, max_width=12):
    """A recognizer with no model loaded - enough to exercise decoding."""
    r = GlinerOnnxRecognizer.__new__(GlinerOnnxRecognizer)
    r.labels, r.threshold, r.max_width = labels, threshold, max_width
    return r


def _logit(p):
    return float(np.log(p / (1 - p)))


def test_token_level_decoding_builds_spans_from_start_end_inside():
    r = _bare_recognizer(["name", "ssn"])
    words = 5  # "Contact John Smith SSN 521-44-3098" -> indices 1-2 name, 4 ssn
    logits = np.full((words, 2, 3), _logit(0.01))
    logits[1, 0] = [_logit(0.9), _logit(0.1), _logit(0.95)]   # John: start, inside
    logits[2, 0] = [_logit(0.1), _logit(0.85), _logit(0.92)]  # Smith: end, inside
    logits[4, 1] = [_logit(0.97), _logit(0.96), _logit(0.98)]  # SSN value
    spans = sorted(r._decode_token_level(logits))
    assert (1, 2, 0) == spans[0][:3] and abs(spans[0][3] - 0.85) < 1e-6  # min(start, end, inside)
    assert (4, 4, 1) == spans[1][:3]


def test_token_level_decoding_rejects_span_with_weak_inside():
    r = _bare_recognizer(["name"])
    logits = np.full((3, 1, 3), _logit(0.01))
    logits[0, 0] = [_logit(0.9), _logit(0.1), _logit(0.9)]
    logits[1, 0] = [_logit(0.1), _logit(0.1), _logit(0.2)]  # gap inside
    logits[2, 0] = [_logit(0.1), _logit(0.9), _logit(0.9)]
    assert all(not (s == 0 and e == 2) for s, e, _c, _p in r._decode_token_level(logits))


def test_predict_greedy_decoding_keeps_best_non_overlapping(monkeypatch):
    r = _bare_recognizer(["name", "location city"])
    r.window_words, r.stride_words = 192, 160
    # Overlapping candidates: "New York" as city (0.9) vs "York" as name (0.6)
    monkeypatch.setattr(r, "_predict_window", lambda words: [(1, 2, 1, 0.9), (2, 2, 0, 0.6)])
    out = r.predict("in New York today")
    assert [(t, round(p, 2)) for _s, _e, t, p in out] == [("LOCATION", 0.9)]
    assert out[0][:2] == (3, 11)  # character offsets of "New York"


def test_calibrated_thresholds_loader(tmp_path, monkeypatch):
    path = tmp_path / "thresholds.json"
    monkeypatch.setattr(engine_mod, "CALIBRATED_THRESHOLDS_PATH", path)
    assert engine_mod._load_calibrated_thresholds() == {}  # missing file -> default
    path.write_text(json.dumps({"per_type": {"PERSON": 0.45}}))
    assert engine_mod._load_calibrated_thresholds() == {"PERSON": 0.45}
    path.write_text("{not json")
    assert engine_mod._load_calibrated_thresholds() == {}  # corrupt -> ignored, not a crash


@pytest.mark.skipif(not os.environ.get("PII_MODEL_DIR"), reason="set PII_MODEL_DIR to run")
def test_gliner_model_end_to_end():
    from app.config import settings
    from app.detection.engine import PiiDetectionEngine

    e = PiiDetectionEngine(settings.spacy_model, pii_model_dir=os.environ["PII_MODEL_DIR"])
    found = {x["text"]: x for x in e.detect("Please contact Sarah Johnson about the invoice.")}
    assert "Sarah Johnson" in found and found["Sarah Johnson"]["selected"]
    skills = e.detect("Skills: Docker, Postman, Kubernetes.")
    assert not any(x["selected"] for x in skills if x["type"] == "PERSON")
