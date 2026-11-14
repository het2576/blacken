"""Span-matching metrics for PII detection.

Several matching schemes are reported because "accuracy" means different
things for a redaction tool than for a generic NER model:

* token  - type-agnostic, per alphanumeric token. The Presidio-research
           standard and the closest proxy for leakage: what fraction of the
           PII tokens in a document end up blacked out. Headline metric.
* overlap - type-agnostic, per entity: a gold entity counts as caught if any
           predicted span overlaps it; a prediction counts as correct if it
           overlaps any gold entity. Does it get flagged at all?
* full_coverage_recall - gold entities whose every alphanumeric character is
           covered. Stricter than overlap: catching "Smith" but not "John"
           still leaks the first name.
* strict - exact boundaries (after trimming whitespace/punctuation) AND the
           right class, one-to-one. The classic seqeval-style NER F1.

F2 weights recall twice as heavily as precision. For PII a miss (a leaked
SSN) costs far more than a false alarm (an extra black bar a reviewer can
untick), so F2 is the right single-number summary; F1 is reported alongside.

Gold spans labelled IGNORE (out-of-scope types) or ORGANIZATION (not
personal data) are "don't care" regions in the core-PII scope: predictions
falling only inside them are neither rewarded nor penalised.
"""

import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from evals.taxonomy import CORE_CLASSES, IGNORE, ORG

Span = Tuple[int, int, str]
_TOKEN = re.compile(r"[A-Za-z0-9]+")
_TRIM = " \t\r\n.,;:!?\"'()[]{}<>|"


def trim(text: str, start: int, end: int) -> Tuple[int, int]:
    while start < end and text[start] in _TRIM:
        start += 1
    while end > start and text[end - 1] in _TRIM:
        end -= 1
    return start, end


def _overlaps(a: Tuple[int, int], b: Tuple[int, int]) -> bool:
    return a[0] < b[1] and b[0] < a[1]


def prf(tp: int, fp: int, fn: int) -> Dict[str, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    f2 = 5 * p * r / (4 * p + r) if (4 * p + r) else 0.0
    return {"precision": p, "recall": r, "f1": f1, "f2": f2, "tp": tp, "fp": fp, "fn": fn}


@dataclass
class DocResult:
    """Raw counts for one document, kept so aggregate scores can be
    bootstrapped at document level."""

    counts: Counter = field(default_factory=Counter)
    per_class: Dict[str, Counter] = field(default_factory=lambda: defaultdict(Counter))
    misses: List[Tuple[str, str]] = field(default_factory=list)  # (class, text)
    false_alarms: List[Tuple[str, str]] = field(default_factory=list)  # (class, text)
    has_gold: bool = False
    has_pred: bool = False


def score_document(
    text: str,
    gold: Sequence[Span],
    pred: Sequence[Span],
    scope: str = "core",
) -> DocResult:
    """`scope` = "core" (all CORE_CLASSES) or "org" (ORGANIZATION only)."""
    in_scope = (lambda c: c in CORE_CLASSES or c not in (IGNORE, ORG)) if scope == "core" else (lambda c: c == ORG)

    gold_t = [(*trim(text, s, e), c) for s, e, c in gold]
    pred_t = [(*trim(text, s, e), c) for s, e, c in pred]
    gold_t = [g for g in gold_t if g[0] < g[1]]
    pred_t = [p for p in pred_t if p[0] < p[1]]

    g_in = [g for g in gold_t if in_scope(g[2])]
    dont_care = [g for g in gold_t if not in_scope(g[2])]
    p_in = [p for p in pred_t if in_scope(p[2])]
    # A prediction that touches no in-scope gold but sits on a don't-care
    # region (e.g. Blacken flags a username as PERSON) is neutral.
    p_scored = [
        p
        for p in p_in
        if any(_overlaps(p, g) for g in g_in) or not any(_overlaps(p, d) for d in dont_care)
    ]

    r = DocResult(has_gold=bool(g_in), has_pred=bool(p_scored))
    c = r.counts

    # ---- token level (type-agnostic) ----
    for m in _TOKEN.finditer(text):
        tok = (m.start(), m.end())
        is_gold = any(_overlaps(tok, g) for g in g_in)
        is_pred = any(_overlaps(tok, p) for p in p_scored)
        if not is_gold and any(_overlaps(tok, d) for d in dont_care):
            continue
        if is_gold and is_pred:
            c["tok_tp"] += 1
        elif is_pred:
            c["tok_fp"] += 1
        elif is_gold:
            c["tok_fn"] += 1

    # ---- entity overlap (type-agnostic) + full coverage ----
    for g in g_in:
        cls = g[2]
        hit = any(_overlaps(g, p) for p in p_scored)
        g_tokens = [(m.start() + g[0], m.end() + g[0]) for m in _TOKEN.finditer(text[g[0] : g[1]])]
        full = bool(g_tokens) and all(any(_overlaps(t, p) for p in p_scored) for t in g_tokens)
        c["ov_gold"] += 1
        c["ov_gold_hit"] += hit
        c["full_cov"] += full
        r.per_class[cls]["gold"] += 1
        r.per_class[cls]["gold_hit"] += hit
        r.per_class[cls]["full_cov"] += full
        if not hit:
            r.misses.append((cls, text[g[0] : g[1]]))
    for p in p_scored:
        correct = any(_overlaps(p, g) for g in g_in)
        c["ov_pred"] += 1
        c["ov_pred_hit"] += correct
        r.per_class[p[2]]["pred"] += 1
        r.per_class[p[2]]["pred_hit"] += correct
        if not correct:
            r.false_alarms.append((p[2], text[p[0] : p[1]]))

    # ---- strict: exact trimmed boundaries + class, one-to-one ----
    unmatched = list(g_in)
    for p in p_scored:
        for i, g in enumerate(unmatched):
            if g[0] == p[0] and g[1] == p[1] and g[2] == p[2]:
                c["strict_tp"] += 1
                r.per_class[g[2]]["strict_tp"] += 1
                unmatched.pop(i)
                break
    c["strict_pred"] += len(p_scored)
    c["strict_gold"] += len(g_in)
    return r


def aggregate(docs: Iterable[DocResult]) -> dict:
    docs = list(docs)
    c: Counter = Counter()
    per_class: Dict[str, Counter] = defaultdict(Counter)
    misses: Counter = Counter()
    false_alarms: Counter = Counter()
    neg_docs = neg_docs_flagged = 0
    for d in docs:
        c.update(d.counts)
        for cls, cc in d.per_class.items():
            per_class[cls].update(cc)
        misses.update(d.misses)
        false_alarms.update(d.false_alarms)
        if not d.has_gold:
            neg_docs += 1
            neg_docs_flagged += d.has_pred

    token = prf(c["tok_tp"], c["tok_fp"], c["tok_fn"])
    # Overlap precision is per prediction and recall is per gold entity (a
    # name split into two predicted fragments is one caught entity and two
    # correct predictions), so they are not derived from a shared TP count.
    overlap = {
        "precision": c["ov_pred_hit"] / c["ov_pred"] if c["ov_pred"] else 0.0,
        "recall": c["ov_gold_hit"] / c["ov_gold"] if c["ov_gold"] else 0.0,
        "missed": c["ov_gold"] - c["ov_gold_hit"],
        "false_alarms": c["ov_pred"] - c["ov_pred_hit"],
    }
    _refresh_f(overlap)
    strict = prf(
        c["strict_tp"], c["strict_pred"] - c["strict_tp"], c["strict_gold"] - c["strict_tp"]
    )

    classes = {}
    for cls, cc in sorted(per_class.items()):
        row = {
            "gold": cc["gold"],
            "pred": cc["pred"],
            "recall": cc["gold_hit"] / cc["gold"] if cc["gold"] else None,
            "full_coverage_recall": cc["full_cov"] / cc["gold"] if cc["gold"] else None,
            "precision": cc["pred_hit"] / cc["pred"] if cc["pred"] else None,
            "strict_recall": cc["strict_tp"] / cc["gold"] if cc["gold"] else None,
        }
        if row["recall"] is not None and row["precision"] is not None:
            p, r = row["precision"], row["recall"]
            row["f1"] = 2 * p * r / (p + r) if p + r else 0.0
            row["f2"] = 5 * p * r / (4 * p + r) if (4 * p + r) else 0.0
        classes[cls] = row

    return {
        "documents": len(docs),
        "gold_entities": c["ov_gold"],
        "predicted_entities": c["ov_pred"],
        "token": token,
        "overlap": overlap,
        "full_coverage_recall": c["full_cov"] / c["ov_gold"] if c["ov_gold"] else 0.0,
        "strict": strict,
        "per_class": classes,
        "negative_documents": neg_docs,
        "negative_false_alarm_rate": (neg_docs_flagged / neg_docs) if neg_docs else None,
        "top_misses": [[cls, t, n] for (cls, t), n in misses.most_common(25)],
        "top_false_alarms": [[cls, t, n] for (cls, t), n in false_alarms.most_common(25)],
    }


def _refresh_f(m: dict) -> None:
    p, r = m["precision"], m["recall"]
    m["f1"] = 2 * p * r / (p + r) if p + r else 0.0
    m["f2"] = 5 * p * r / (4 * p + r) if (4 * p + r) else 0.0


def bootstrap_ci(
    docs: Sequence[DocResult],
    n_resamples: int = 1000,
    seed: int = 7,
    alpha: float = 0.05,
) -> Dict[str, Tuple[float, float]]:
    """Document-level percentile bootstrap for the headline metrics. Resamples
    whole documents (not tokens), because errors within a document are
    correlated - resampling tokens would give falsely narrow intervals."""
    rng = random.Random(seed)
    keys = ("tok_tp", "tok_fp", "tok_fn", "ov_gold", "ov_gold_hit", "ov_pred", "ov_pred_hit")
    rows = [[d.counts[k] for k in keys] for d in docs]
    n = len(rows)
    if n == 0:
        return {}
    stats: Dict[str, List[float]] = defaultdict(list)
    for _ in range(n_resamples):
        tot = [0] * len(keys)
        for _ in range(n):
            row = rows[rng.randrange(n)]
            for i, v in enumerate(row):
                tot[i] += v
        t = dict(zip(keys, tot))
        tok = prf(t["tok_tp"], t["tok_fp"], t["tok_fn"])
        stats["token_precision"].append(tok["precision"])
        stats["token_recall"].append(tok["recall"])
        stats["token_f1"].append(tok["f1"])
        stats["token_f2"].append(tok["f2"])
        ovr = t["ov_gold_hit"] / t["ov_gold"] if t["ov_gold"] else 0.0
        ovp = t["ov_pred_hit"] / t["ov_pred"] if t["ov_pred"] else 0.0
        stats["overlap_recall"].append(ovr)
        stats["overlap_precision"].append(ovp)
    out = {}
    for k, vals in stats.items():
        vals.sort()
        lo = vals[int((alpha / 2) * len(vals))]
        hi = vals[min(len(vals) - 1, int((1 - alpha / 2) * len(vals)))]
        out[k] = (lo, hi)
    return out


def select_predictions(entities: List[dict], operating_point: str) -> List[dict]:
    """`detected` = everything the reviewer is shown; `auto` = what is
    pre-ticked and gets redacted if the user just clicks Redact."""
    if operating_point == "auto":
        return [e for e in entities if e.get("selected")]
    return list(entities)


def ci_str(ci: Optional[Tuple[float, float]]) -> str:
    return f"[{ci[0]*100:.1f}, {ci[1]*100:.1f}]" if ci else ""
