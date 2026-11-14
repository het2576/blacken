"""Latency, throughput and memory benchmark.

    python -m evals.run_latency                      # local engine + API
    python -m evals.run_latency --live-url https://blacken-production.up.railway.app --live-n 10

Local: model load time, peak RSS, detection latency vs document size, and
full upload->redact->download latency for a multi-page PDF.
Live: sequential (never concurrent) requests against a deployed backend,
using synthetic documents only; every document is deleted right after.
"""

import argparse
import io
import json
import resource
import statistics
import sys
import time
import uuid
from pathlib import Path

from evals.datasets import load_blacken_gold
from evals.run_e2e import build_pdf

RESULTS_DIR = Path(__file__).resolve().parent / "results"
LIVE_DOC = (
    "Patient Name: Robert Johnson\nDOB: 11/22/1958\nSSN: 234-56-7890\n"
    "Email: robert.johnson@example.com\nPhone: (617) 394-8821\n"
    "Address: 42 Wallaby Way, Springfield\nRoutine checkup with no other findings.\n"
)


def _pct(values, q):
    s = sorted(values)
    return s[min(len(s) - 1, int(q * (len(s) - 1)))]


def _summary(values):
    return {
        "n": len(values),
        "p50_ms": statistics.median(values),
        "p95_ms": _pct(values, 0.95),
        "max_ms": max(values),
    }


def _rss_mb() -> float:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / (1024 * 1024) if sys.platform == "darwin" else rss / 1024


def bench_local(model: str, repeats: int):
    rss_before = _rss_mb()
    t0 = time.perf_counter()
    from app.detection.engine import PiiDetectionEngine

    engine = PiiDetectionEngine(spacy_model=model)
    load_s = time.perf_counter() - t0
    engine.detect("warm up John Smith 234-56-7890")
    rss_after = _rss_mb()

    corpus = "\n\n".join(s.text for s in load_blacken_gold())
    by_size = {}
    for size in (500, 2_000, 10_000, 50_000):
        text = (corpus * (size // len(corpus) + 1))[:size]
        times = []
        for _ in range(repeats):
            t = time.perf_counter()
            engine.detect(text)
            times.append((time.perf_counter() - t) * 1000)
        s = _summary(times)
        s["chars_per_sec"] = size / (s["p50_ms"] / 1000)
        by_size[str(size)] = s
        print(f"detect {size:>6} chars  p50={s['p50_ms']:7.1f}ms  p95={s['p95_ms']:7.1f}ms  {s['chars_per_sec']:,.0f} chars/s")

    return engine, {
        "model": Path(model).name,
        "model_load_s": load_s,
        "rss_before_mb": rss_before,
        "peak_rss_after_load_mb": rss_after,
        "detect_by_chars": by_size,
    }


def bench_local_api(pages: int, repeats: int):
    import tempfile

    from fastapi.testclient import TestClient

    from app.config import settings

    tmp = Path(tempfile.mkdtemp(prefix="blacken-lat-"))
    settings.upload_dir, settings.data_dir = tmp / "u", tmp / "d"
    settings.db_path = tmp / "d" / "lat.db"
    settings.upload_dir.mkdir()
    settings.data_dir.mkdir()
    from app.main import app

    gold = [s.text for s in load_blacken_gold() if not s.id.startswith("negative")]
    text = "\n\n".join((gold * 10)[: pages * 2])
    pdf = build_pdf(text)
    times = []
    with TestClient(app, headers={"X-Owner-Id": f"lat-{uuid.uuid4()}"}) as c:
        for _ in range(repeats):
            t = time.perf_counter()
            up = c.post("/api/upload", files={"file": ("bench.pdf", pdf, "application/pdf")}).json()
            c.post("/api/redact", json={"document_id": up["document_id"]})
            c.get(f"/api/download/{up['document_id']}")
            times.append((time.perf_counter() - t) * 1000)
            c.delete(f"/api/document/{up['document_id']}")
    import pymupdf

    n_pages = pymupdf.open(stream=pdf, filetype="pdf").page_count
    s = _summary(times)
    s["pages"] = n_pages
    print(f"API pdf {n_pages} pages upload+redact+download  p50={s['p50_ms']:.0f}ms p95={s['p95_ms']:.0f}ms")
    return s


def bench_live(url: str, n: int):
    import httpx

    owner = {"X-Owner-Id": f"eval-latency-{uuid.uuid4()}"}
    pdf = build_pdf(LIVE_DOC)
    out = {"url": url, "health": [], "upload_detect": [], "redact": [], "download": [], "errors": 0}
    with httpx.Client(base_url=url, timeout=60) as c:
        t = time.perf_counter()
        h = c.get("/api/health")
        out["first_health_ms"] = (time.perf_counter() - t) * 1000  # may include cold start
        out["deployed_features"] = h.json().get("features", {}) if h.status_code == 200 else {}
        for _ in range(n):
            t = time.perf_counter()
            c.get("/api/health")
            out["health"].append((time.perf_counter() - t) * 1000)

            t = time.perf_counter()
            r = c.post("/api/upload", files={"file": ("eval.pdf", pdf, "application/pdf")}, headers=owner)
            out["upload_detect"].append((time.perf_counter() - t) * 1000)
            if r.status_code != 200:
                out["errors"] += 1
                continue
            body = r.json()
            doc_id = body["document_id"]
            out.setdefault("entities_detected", body["entity_count"])
            t = time.perf_counter()
            rr = c.post("/api/redact", json={"document_id": doc_id}, headers=owner)
            out["redact"].append((time.perf_counter() - t) * 1000)
            t = time.perf_counter()
            rd = c.get(f"/api/download/{doc_id}", headers=owner)
            out["download"].append((time.perf_counter() - t) * 1000)
            out["errors"] += (rr.status_code != 200) + (rd.status_code != 200)
            c.delete(f"/api/document/{doc_id}", headers=owner)
            time.sleep(0.5)  # be gentle with a free-tier host
    for k in ("health", "upload_detect", "redact", "download"):
        if out[k]:
            out[k] = _summary(out[k])
            print(f"live {k:14} p50={out[k]['p50_ms']:7.0f}ms  p95={out[k]['p95_ms']:7.0f}ms")
    print(f"live first request (cold start incl.): {out['first_health_ms']:.0f}ms  errors={out['errors']}")
    return out


def main(argv=None):
    from evals.run_detection import DEFAULT_MODEL

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument("--pdf-pages", type=int, default=10)
    ap.add_argument("--live-url", default=None)
    ap.add_argument("--live-n", type=int, default=10)
    ap.add_argument("--skip-local", action="store_true")
    args = ap.parse_args(argv)

    # Local and live results go to separate files so a live-only run never
    # overwrites the local numbers (evals.report reads both).
    RESULTS_DIR.mkdir(exist_ok=True)
    if not args.skip_local:
        local = {}
        _, local["local"] = bench_local(args.model, args.repeats)
        local["local_api_pdf"] = bench_local_api(args.pdf_pages, max(3, args.repeats // 4))
        out = RESULTS_DIR / "latency.json"
        out.write_text(json.dumps(local, indent=2))
        print(f"wrote {out}")
    if args.live_url:
        live = {"live": bench_live(args.live_url.rstrip("/"), args.live_n)}
        out = RESULTS_DIR / "latency_live.json"
        out.write_text(json.dumps(live, indent=2))
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
