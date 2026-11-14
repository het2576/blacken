"""Detection-quality evaluation.

    python -m evals.run_detection                       # all datasets, test splits
    python -m evals.run_detection --datasets blacken_gold --split test
    python -m evals.run_detection --split dev --limit 3000   # tuning runs

Writes evals/results/detection_<split>.json and prints a summary. Each
dataset is scored at two operating points:

  detected - every entity shown to the reviewer
  auto     - only entities pre-ticked for redaction (what gets blacked out if
             the user just presses Redact) - the number that matters most
"""

import argparse
import json
import os
import statistics
import time
from collections import defaultdict
from pathlib import Path

from evals import datasets as ds
from evals.metrics import aggregate, bootstrap_ci, score_document, select_predictions
from evals.taxonomy import engine_class

RESULTS_DIR = Path(__file__).resolve().parent / "results"
DEFAULT_MODEL = os.environ.get(
    "SPACY_MODEL", str(Path(__file__).resolve().parent.parent / "models" / "en_core_web_lg_pruned")
)
ALL_DATASETS = ("blacken_gold", "ai4privacy", "gretel")


def to_spans(entities):
    return [(e["start"], e["end"], engine_class(e["type"])) for e in entities]


def evaluate_dataset(engine, samples, bootstrap: int = 1000, group_key: str = None):
    docs = {"detected": [], "auto": []}
    org_docs = {"detected": [], "auto": []}
    groups = defaultdict(lambda: {"detected": [], "auto": []})
    latencies_ms, chars = [], 0

    for s in samples:
        t0 = time.perf_counter()
        entities = engine.detect(s.text)
        latencies_ms.append((time.perf_counter() - t0) * 1000)
        chars += len(s.text)
        for op in ("detected", "auto"):
            pred = to_spans(select_predictions(entities, op))
            r = score_document(s.text, s.spans, pred, scope="core")
            docs[op].append(r)
            org_docs[op].append(score_document(s.text, s.spans, pred, scope="org"))
            if group_key:
                groups[s.meta.get(group_key, "?")][op].append(r)

    out = {}
    for op in ("detected", "auto"):
        res = aggregate(docs[op])
        res["ci95"] = bootstrap_ci(docs[op], n_resamples=bootstrap) if bootstrap else {}
        org = aggregate(org_docs[op])
        res["organization"] = {
            "gold": org["gold_entities"],
            "recall": org["overlap"]["recall"],
            "precision": org["overlap"]["precision"],
        }
        if group_key:
            res["by_group"] = {
                g: {
                    "documents": len(v[op]),
                    "token_recall": (a := aggregate(v[op]))["token"]["recall"],
                    "token_precision": a["token"]["precision"],
                    "entity_recall": a["overlap"]["recall"],
                }
                for g, v in sorted(groups.items())
            }
        out[op] = res

    lat_sorted = sorted(latencies_ms)
    out["latency"] = {
        "docs": len(latencies_ms),
        "mean_chars": chars / max(1, len(samples)),
        "p50_ms": statistics.median(lat_sorted),
        "p95_ms": lat_sorted[int(0.95 * (len(lat_sorted) - 1))],
        "p99_ms": lat_sorted[int(0.99 * (len(lat_sorted) - 1))],
        "chars_per_sec": chars / (sum(latencies_ms) / 1000) if latencies_ms else 0,
    }
    return out


def fmt(x):
    return "  -  " if x is None else f"{x*100:5.1f}"


def print_summary(name, res):
    print(f"\n=== {name} ===  docs={res['auto']['documents']}  gold={res['auto']['gold_entities']}")
    print(f"{'op':9} {'tokP':>6} {'tokR':>6} {'tokF1':>6} {'tokF2':>6} | {'entP':>6} {'entR':>6} {'fullR':>6} | {'strictF1':>8} | negFA")
    for op in ("detected", "auto"):
        r = res[op]
        t, o, s = r["token"], r["overlap"], r["strict"]
        fa = r["negative_false_alarm_rate"]
        print(
            f"{op:9} {fmt(t['precision'])}  {fmt(t['recall'])}  {fmt(t['f1'])}  {fmt(t['f2'])} | "
            f"{fmt(o['precision'])}  {fmt(o['recall'])}  {fmt(r['full_coverage_recall'])} | "
            f"{fmt(s['f1']):>8} | {fmt(fa) if fa is not None else '  -'}"
        )
    print("  per-class (auto): " + ", ".join(
        f"{c} R={fmt(v['recall']).strip()} P={fmt(v['precision']).strip()} (n={v['gold']})"
        for c, v in res["auto"]["per_class"].items()
    ))
    lat = res["latency"]
    print(f"  latency p50={lat['p50_ms']:.1f}ms p95={lat['p95_ms']:.1f}ms  ({lat['chars_per_sec']:.0f} chars/s, mean {lat['mean_chars']:.0f} chars/doc)")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=list(ALL_DATASETS), choices=ALL_DATASETS)
    ap.add_argument("--split", default="test", choices=("dev", "test"))
    ap.add_argument("--limit", type=int, default=None, help="random subsample per dataset (seeded)")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--tag", default="", help="suffix for the results file, e.g. 'baseline'")
    ap.add_argument("--pii-model", default=os.environ.get("PII_MODEL_DIR"), help="GLiNER ONNX dir")
    ap.add_argument("--pii-threshold", type=float, default=0.5)
    ap.add_argument("--no-spacy-ner", action="store_true")
    ap.add_argument("--no-calibration", action="store_true", help="ignore app/detection/thresholds.json")
    args = ap.parse_args(argv)

    from app.detection.engine import PiiDetectionEngine

    engine = PiiDetectionEngine(
        spacy_model=args.model,
        pii_model_dir=args.pii_model,
        pii_model_threshold=args.pii_threshold,
        use_spacy_ner=not args.no_spacy_ner,
        use_calibrated_thresholds=not args.no_calibration,
    )
    results = {
        "split": args.split,
        "model": Path(args.model).name,
        "pii_model": Path(args.pii_model).name if args.pii_model else None,
        "pii_threshold": args.pii_threshold if args.pii_model else None,
        "spacy_ner": not args.no_spacy_ner,
        "calibrated_thresholds": bool(engine.auto_select_thresholds),
        "datasets": {},
    }
    group_keys = {"blacken_gold": "domain", "gretel": "document_type", "ai4privacy": "locale"}

    for name in args.datasets:
        if name == "blacken_gold" and args.split == "dev":
            print("[evals] blacken_gold is test-only; skipping on dev split")
            continue
        samples = ds.load(name, split=args.split, limit=args.limit)
        res = evaluate_dataset(engine, samples, bootstrap=args.bootstrap, group_key=group_keys[name])
        results["datasets"][name] = res
        print_summary(name, res)

    RESULTS_DIR.mkdir(exist_ok=True)
    suffix = f"_{args.tag}" if args.tag else ""
    out_path = RESULTS_DIR / f"detection_{args.split}{suffix}.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
