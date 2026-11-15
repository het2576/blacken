"""Build evals/RESULTS.md from the JSON files in evals/results/.

    python -m evals.report

Every number in RESULTS.md comes from a results file, so the report is
reproducible: re-run the harness, re-run this, diff.
"""

import json
from pathlib import Path

RESULTS = Path(__file__).resolve().parent / "results"
OUT = Path(__file__).resolve().parent / "RESULTS.md"
NAMES = {
    "blacken_gold": "Blacken gold (hand-annotated)",
    "ai4privacy": "ai4privacy pii-masking-400k (EN val)",
    "gretel": "Gretel synthetic PII finance (EN test)",
}


def _load(name):
    p = RESULTS / name
    return json.loads(p.read_text()) if p.exists() else None


def pct(x, d=1):
    return "–" if x is None else f"{x*100:.{d}f}%"


def ci(r, key):
    c = r.get("ci95", {}).get(key)
    return f" [{c[0]*100:.1f}–{c[1]*100:.1f}]" if c else ""


def detection_table(res, op="auto"):
    rows = [
        "| Dataset | Docs | Gold entities | Token P | Token R (95% CI) | Token F1 | Token F2 | Entity P | Entity R | Strict F1 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, d in res["datasets"].items():
        r = d[op]
        rows.append(
            f"| {NAMES.get(name, name)} | {r['documents']:,} | {r['gold_entities']:,} | "
            f"{pct(r['token']['precision'])} | {pct(r['token']['recall'])}{ci(r, 'token_recall')} | "
            f"{pct(r['token']['f1'])} | {pct(r['token']['f2'])} | {pct(r['overlap']['precision'])} | "
            f"{pct(r['overlap']['recall'])} | {pct(r['strict']['f1'])} |"
        )
    return "\n".join(rows)


def delta_table(before, after, op="auto"):
    rows = [
        "| Dataset | Metric | Before | After | Δ |",
        "|---|---|---:|---:|---:|",
    ]
    for name in after["datasets"]:
        if name not in before["datasets"]:
            continue
        b, a = before["datasets"][name][op], after["datasets"][name][op]
        for label, get in (
            ("Token recall", lambda r: r["token"]["recall"]),
            ("Token precision", lambda r: r["token"]["precision"]),
            ("Token F2", lambda r: r["token"]["f2"]),
            ("Entity recall", lambda r: r["overlap"]["recall"]),
            ("Strict F1", lambda r: r["strict"]["f1"]),
        ):
            vb, va = get(b), get(a)
            rows.append(
                f"| {NAMES.get(name, name)} | {label} | {pct(vb)} | {pct(va)} | {(va - vb) * 100:+.1f} pp |"
            )
    return "\n".join(rows)


def per_class_table(res, dataset, op="auto"):
    pc = res["datasets"][dataset][op]["per_class"]
    rows = ["| Class | Gold | Recall | Full-coverage recall | Precision |", "|---|---:|---:|---:|---:|"]
    for cls, v in pc.items():
        if v["gold"]:
            rows.append(
                f"| {cls} | {v['gold']} | {pct(v['recall'])} | {pct(v['full_coverage_recall'])} | {pct(v['precision'])} |"
            )
    return "\n".join(rows)


def e2e_table(before, after):
    rows = [
        "| Format | Leak rate before | Leak rate after | Redactor-only leak (oracle) before → after | Over-redaction before → after | Metadata leaks before → after |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for fmt, a in after["formats"].items():
        b = before["formats"].get(fmt, {}) if before else {}
        rows.append(
            f"| {fmt} | {pct(b.get('leak_rate_auto'))} | **{pct(a['leak_rate_auto'])}** | "
            f"{pct(b.get('leak_rate_oracle'))} → {pct(a['leak_rate_oracle'])} | "
            f"{pct(b.get('over_redaction_rate'))} → {pct(a['over_redaction_rate'])} | "
            f"{b.get('metadata_leak_docs', '–')}/{b.get('documents', '–')} → {a['metadata_leak_docs']}/{a['documents']} |"
        )
    return "\n".join(rows)


def _get(res, dataset, *path):
    try:
        r = res["datasets"][dataset]["auto"]
        for p in path:
            r = r[p]
        return r
    except (KeyError, TypeError):
        return None


def names_update_section():
    """Name-recognition work on the deployed profile (sm + rules + GLiNER-edge):
    before = code at the previous commit, v1 = name rules frozen before the
    held-out set was written, final = after the engine bug fixes."""
    before = _load("detection_test_names_before_F.json")  # indian suites, old code
    v1 = _load("detection_test_names_v1_F.json")  # held-out, frozen rules
    final = _load("detection_test_names_final_F.json")
    old_main = _load("detection_test_final_F.json")  # gold + public, old code
    pub_new = _load("detection_test_names_final_F_public.json")
    if not (before and final):
        return []
    rows = ["| Test set | Metric | Before | After |", "|---|---|---:|---:|"]

    def row(label, metric, b, a):
        rows.append(f"| {label} | {metric} | {pct(b)} | **{pct(a)}** |")

    row("Indian-format docs (67 incl. ALL-CAPS twins)", "Name recall",
        _get(before, "indian_gold", "per_class", "PERSON", "recall"), _get(final, "indian_gold", "per_class", "PERSON", "recall"))
    row("Indian-format docs", "Name precision",
        _get(before, "indian_gold", "per_class", "PERSON", "precision"), _get(final, "indian_gold", "per_class", "PERSON", "precision"))
    hb = _get(before, "indian_heldout", "per_class", "PERSON", "recall")
    hv = _get(v1, "indian_heldout", "per_class", "PERSON", "recall") if v1 else None
    hf = _get(final, "indian_heldout", "per_class", "PERSON", "recall")
    rows.append(f"| **Held-out Indian names (unseen, 35 docs)** | Name recall | {pct(hb)} | **{pct(hv)}** frozen rules → {pct(hf)} after bug fixes* |")
    if old_main:
        for metric, path in (("Token recall", ("token", "recall")), ("Token precision", ("token", "precision")),
                             ("Token F2", ("token", "f2")), ("Name recall", ("per_class", "PERSON", "recall")),
                             ("Strict F1", ("strict", "f1"))):
            row("Blacken gold (44 docs)", metric, _get(old_main, "blacken_gold", *path), _get(final, "blacken_gold", *path))
        if pub_new:
            for ds in ("ai4privacy", "gretel"):
                for metric, path in (("Token recall", ("token", "recall")), ("Token precision", ("token", "precision"))):
                    row(NAMES.get(ds, ds), metric, _get(old_main, ds, *path), _get(pub_new, ds, *path))
    out = ["## Name recognition update — deployed profile (small spaCy + rules + GLiNER-edge)", "",
           "Real uploads showed Indian names, ALL-CAPS forms and resume headers being missed, and first names "
           "labelled LOCATION/ORGANIZATION. Fixes: engine bugs (Presidio de-duplication erasing exact name matches, "
           "overlapping detectors cutting names short, generic date guesses beating validated IDs, name runs "
           "crossing sentence ends), an ALL-CAPS re-read for the NER models, Indian name / honorific / relation "
           "(S/o, W/o) / field-label recognizers (`app/detection/indian_names.py`), type correction and tech-term "
           "filtering. Auto-selected entities.", "", *rows, "",
           "\\* The held-out set (`evals/data/indian_heldout.txt`) was written after the name rules were frozen, with "
           "names absent from the name lists (1 of 81 name words overlaps). The frozen-rules number is the clean "
           "generalisation estimate; the final number is slightly optimistic because two engine bugs were found by "
           "debugging held-out misses and one field label it contains was added.", ""]
    if old_main and pub_new:
        lat = []
        for ds in ("ai4privacy", "gretel"):
            a, b = old_main["datasets"].get(ds, {}).get("latency"), pub_new["datasets"].get(ds, {}).get("latency")
            if a and b:
                lat.append(f"{NAMES.get(ds, ds)} (mean {b['mean_chars']:.0f} chars): p50 {a['p50_ms']:.0f} → {b['p50_ms']:.0f} ms, "
                           f"p95 {a['p95_ms']:.0f} → {b['p95_ms']:.0f} ms")
        if lat:
            out += ["Speed cost (detection per document, Apple Silicon): " + "; ".join(lat) + ". The ALL-CAPS re-read "
                    "runs the NER models a second time on documents with capitalised stretches, so long documents pay most.", ""]
    e_old, e_new = _load("e2e_final_F.json"), _load("e2e_names_final_F.json")
    if e_old and e_new:
        out += ["End-to-end leakage (gold documents as real files), before → after:", "",
                "| Format | Leak rate | Over-redaction |", "|---|---:|---:|"]
        for fmt, n in e_new["formats"].items():
            o = e_old["formats"].get(fmt, {})
            out.append(f"| {fmt} | {pct(o.get('leak_rate_auto'))} → **{pct(n['leak_rate_auto'])}** | "
                       f"{pct(o.get('over_redaction_rate'))} → {pct(n['over_redaction_rate'])} |")
        out.append("")
    prof = [("Deployed: small spaCy + rules + GLiNER-edge", "names_final_F", "F_sm_edge_ner"),
            ("Pruned lg spaCy + rules + GLiNER-edge", "names_final_B", "B_lgpruned_edge"),
            ("Pruned lg spaCy + rules (no transformer)", "names_final_A", None)]
    mem = (_load("memory_request.json") or {}).get("configs", {})
    prow = ["| Profile | Gold R / P | Held-out names R | Indian names R | Peak memory, scanned-PDF request |",
            "|---|---:|---:|---:|---:|"]
    for label, tag, mkey in prof:
        r = _load(f"detection_test_{tag}.json")
        if not r:
            continue
        m = mem.get(mkey or "", {})
        peak = f"{m['peak_scanned_5p_mb']} MB" if m else "–"
        prow.append(f"| {label} | {pct(_get(r, 'blacken_gold', 'token', 'recall'))} / {pct(_get(r, 'blacken_gold', 'token', 'precision'))} | "
                    f"{pct(_get(r, 'indian_heldout', 'per_class', 'PERSON', 'recall'))} | "
                    f"{pct(_get(r, 'indian_gold', 'per_class', 'PERSON', 'recall'))} | {peak} |")
    if len(prow) > 2:
        out += ["Profiles on the final code. Memory is the whole app plus Tesseract during a scanned-PDF upload "
                "(`results/memory_request.json`) — the number a container limit sees. Pruned-lg + GLiNER is ~1–1.5 pts "
                "more accurate but peaks ~640 MB, so it does not fit a 512 MB host.", "", *prow, ""]
    return out


def main():
    final = _load("detection_test_final.json")
    base_gold = _load("detection_test_baseline.json")
    base_pub = _load("detection_test_baseline_public.json")
    e2e_b, e2e_a = _load("e2e_baseline.json"), _load("e2e_final.json")
    live = _load("latency_live.json")
    local = _load("latency.json")

    before = {"datasets": {**(base_gold or {}).get("datasets", {}), **(base_pub or {}).get("datasets", {})}}
    parts = ["# Blacken evaluation results", "",
             "Generated by `python -m evals.report` from `evals/results/*.json`. "
             "Methodology and caveats: `evals/README.md`.", ""]
    parts += names_update_section()

    if final:
        parts += ["## Detection quality — production model, auto-selected entities (one-click redaction)", "",
                  detection_table(final, "auto"), "",
                  "## Detection quality — everything shown to the reviewer", "",
                  detection_table(final, "detected"), ""]
        gold_fa = final["datasets"].get("blacken_gold", {}).get("auto", {}).get("negative_false_alarm_rate")
        if gold_fa is not None:
            parts += [f"False-alarm rate on PII-free gold documents (auto): **{pct(gold_fa)}**", ""]
        parts += ["## Before → after (auto-selected)", "", delta_table(before, final), ""]
        if "blacken_gold" in final["datasets"]:
            parts += ["## Per-class (gold set, auto-selected)", "", per_class_table(final, "blacken_gold"), ""]
    depl, depl_e2e = _load("detection_test_final_F.json"), _load("e2e_final_F.json")
    if depl and final:
        parts += ["## Deployed profile (small spaCy + rules + GLiNER-PII) — held-out test splits, auto-selected", "",
                  "Same full test splits as the default profile above; what runs on the 512 MB Railway host.", "",
                  detection_table(depl, "auto"), "",
                  "Change vs the default profile (pruned spaCy lg + rules):", "",
                  delta_table(final, depl), ""]
        if depl_e2e and e2e_a:
            rows = ["| Format | Leak (default → deployed) | Redactor-only leak | Over-redaction (default → deployed) |",
                    "|---|---:|---:|---:|"]
            for fmt, d in depl_e2e["formats"].items():
                a = e2e_a["formats"].get(fmt)
                if a:
                    rows.append(f"| {fmt} | {pct(a['leak_rate_auto'])} → **{pct(d['leak_rate_auto'])}** | "
                                f"{pct(d['leak_rate_oracle'])} | {pct(a['over_redaction_rate'])} → {pct(d['over_redaction_rate'])} |")
            parts += ["End-to-end leakage with the deployed profile:", "", *rows, ""]
    memory = (_load("memory.json") or {}).get("configs", {})
    cfg_rows = []
    for tag, mem in memory.items():
        dev, gold = _load(f"detection_dev_cfg_{tag}.json"), _load(f"detection_test_cfg_{tag}.json")
        if not dev or not gold:
            continue

        def rp(res, name):
            r = res["datasets"][name]["auto"]["token"]
            return f"{pct(r['recall'])} / {pct(r['precision'])}"

        # Request processing needs headroom on top of the engine: PDF parsing
        # and 300-DPI page rasters (~26 MB/page) in-process, plus Tesseract
        # as a subprocess. <=380 MB leaves ~130 MB; <=450 MB is workable with
        # MAX_CONCURRENT_JOBS=1.
        fits = "yes" if mem["peak_mb"] <= 380 else ("tight (1 job at a time)" if mem["peak_mb"] <= 450 else "no")
        cfg_rows.append(
            f"| {mem['label']} | {rp(dev, 'ai4privacy')} | {rp(dev, 'gretel')} | {rp(gold, 'blacken_gold')} | "
            f"{mem['steady_mb']} / {mem['peak_mb']} MB | {mem['ms_per_10k_chars']} ms | {fits} |"
        )
    if cfg_rows:
        parts += [
            "## Model configuration comparison (auto-selected token recall / precision)", "",
            "Dev splits: 1,000 docs each (seeded); gold: all 44 docs. Memory = detection engine, steady / peak. "
            "Memory is macOS physical footprint (closest analogue of Linux RSS). 'Fits 512 MB' accounts for request-processing headroom (OCR rasters + Tesseract subprocess).", "",
            "| Configuration | ai4privacy dev | Gretel dev | Gold test | Memory | Speed / 10k chars | Fits 512 MB |",
            "|---|---:|---:|---:|---:|---:|---|",
            *cfg_rows, "",
        ]
    if e2e_a:
        parts += ["## End-to-end redaction leakage (gold documents rendered to real files)", "",
                  e2e_table(e2e_b, e2e_a), ""]
    ocr300, ocr200 = _load("e2e_ocr_v2.json"), _load("e2e_ocr_dpi200.json")
    if ocr300 and e2e_a:
        parts += ["## OCR changes (grayscale + autocontrast preprocessing, keep low-confidence words; DPI tested)", "",
                  "Leak with default redaction / redactor-only (oracle) leak. Images are processed at native "
                  "resolution, so DPI only affects scanned PDFs. Shipped: 200 DPI + preprocessing.", "",
                  "| Format | Before (200 DPI, conf ≥ 30) | 300 DPI + preprocessing | 200 DPI + preprocessing (shipped) |",
                  "|---|---:|---:|---:|"]
        for fmt in ("pdf_scanned", "png"):
            b, a3 = e2e_a["formats"].get(fmt), ocr300["formats"].get(fmt)
            a2 = (ocr200 or {}).get("formats", {}).get(fmt) if fmt == "pdf_scanned" else a3
            cell = lambda r: f"{pct(r['leak_rate_auto'])} / {pct(r['leak_rate_oracle'])}" if r else "–"
            parts.append(f"| {fmt} | {cell(b)} | {cell(a3)} | **{cell(a2)}** |")
        parts.append("")
    cal = _load("rejected/detection_test_calibrated.json")
    if cal and final:
        parts += ["## Threshold calibration — evaluated and rejected", "",
                  "Per-type auto-select thresholds fitted on dev splits (`evals/calibrate.py`, token F2), "
                  "then validated on test. Rejected: public-set gains come largely from learning *not* to flag "
                  "what those datasets leave unlabelled (e.g. dates → 0.85), and the only fully annotated set "
                  "shows that costs real recall. Fit archived in `results/rejected/`.", "",
                  "| Test set | Default R / P / F2 | Calibrated R / P / F2 |", "|---|---:|---:|"]
        for name, d in final["datasets"].items():
            c = cal["datasets"].get(name)
            if c:
                a, b = d["auto"]["token"], c["auto"]["token"]
                parts.append(
                    f"| {NAMES.get(name, name)} | {pct(a['recall'])} / {pct(a['precision'])} / {pct(a['f2'])} | "
                    f"{pct(b['recall'])} / {pct(b['precision'])} / {pct(b['f2'])} |"
                )
        parts.append("")
    if local and "local" in local:
        l = local["local"]
        parts += [f"## Latency & memory (local, Apple Silicon, single process, {l.get('model', 'default model')})", "",
                  f"- Model load: {l['model_load_s']:.1f}s; peak RSS after load: {l['peak_rss_after_load_mb']:.0f} MB", ""]
        parts += ["| Document size | p50 | p95 | Throughput |", "|---:|---:|---:|---:|"]
        for size, s in l["detect_by_chars"].items():
            parts.append(f"| {int(size):,} chars | {s['p50_ms']:.0f} ms | {s['p95_ms']:.0f} ms | {s['chars_per_sec']:,.0f} chars/s |")
        if "local_api_pdf" in local:
            a = local["local_api_pdf"]
            parts.append(f"\nFull upload→detect→redact→download, {a['pages']}-page PDF: p50 {a['p50_ms']:.0f} ms, p95 {a['p95_ms']:.0f} ms")
        parts.append("")
    if live and "live" in live:
        lv = live["live"]
        feats = lv.get("deployed_features") or {}
        profile = ""
        if feats:
            profile = ("Deployed profile: " + ("spaCy + rules + GLiNER-PII transformer" if feats.get("transformer_model")
                       else "spaCy + rules") + ("; encryption at rest on" if feats.get("encryption_at_rest") else "")
                       + (f"; {lv['entities_detected']} entities detected in the benchmark PDF" if "entities_detected" in lv else "")
                       + ".")
        parts += [f"## Live production ({lv['url']}, {lv['health']['n']} sequential requests)", "",
                  *([profile, ""] if profile else []),
                  "| Call | p50 | p95 |", "|---|---:|---:|"]
        for k in ("health", "upload_detect", "redact", "download"):
            parts.append(f"| {k} | {lv[k]['p50_ms']:.0f} ms | {lv[k]['p95_ms']:.0f} ms |")
        parts += ["", f"Cold start (first request): {lv['first_health_ms']/1000:.1f}s · errors: {lv['errors']}. "
                  "`health` p50 is network round-trip; server-side processing ≈ upload p50 − health p50.", ""]

    OUT.write_text("\n".join(parts))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
