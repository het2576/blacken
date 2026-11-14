"""Dataset loaders. Every loader yields `Sample`s in one shared format, with
gold spans already mapped onto evals.taxonomy classes.

Splits are explicit and matter: anything used to tune the engine must come
from a `dev` split, and headline numbers must come from `test` splits that
were never looked at while tuning. Public datasets are downloaded on first use
into evals/.cache/ (gitignored) - nothing third-party is committed.
"""

import json
import random
import re
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

from evals.taxonomy import AI4PRIVACY_TO_CLASS, GRETEL_TO_CLASS, IGNORE

EVALS_DIR = Path(__file__).resolve().parent
CACHE_DIR = EVALS_DIR / ".cache"
DATA_DIR = EVALS_DIR / "data"

HF = "https://huggingface.co/datasets"
SOURCES = {
    ("ai4privacy", "dev"): f"{HF}/ai4privacy/pii-masking-400k/resolve/main/data/train/1en.jsonl",
    ("ai4privacy", "test"): f"{HF}/ai4privacy/pii-masking-400k/resolve/main/data/validation/1en.jsonl",
    ("gretel", "dev"): f"{HF}/gretelai/synthetic_pii_finance_multilingual/resolve/main/data/English_train-00000-of-00001.parquet",
    ("gretel", "test"): f"{HF}/gretelai/synthetic_pii_finance_multilingual/resolve/main/data/English_test-00000-of-00001.parquet",
}

GoldSpan = Tuple[int, int, str]  # (start, end, class)


@dataclass
class Sample:
    id: str
    text: str
    spans: List[GoldSpan]
    dataset: str
    meta: dict = field(default_factory=dict)


def _download(dataset: str, split: str) -> Path:
    url = SOURCES[(dataset, split)]
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"{dataset}_{split}{Path(url).suffix}"
    if not path.exists() or path.stat().st_size == 0:
        print(f"[evals] downloading {dataset}/{split} -> {path.name}")
        tmp = path.with_suffix(path.suffix + ".part")
        # python.org macOS builds ship without system CA certs; prefer
        # certifi's bundle when present so downloads work out of the box.
        try:
            import ssl

            import certifi

            ctx = ssl.create_default_context(cafile=certifi.where())
        except ImportError:
            ctx = None
        with urllib.request.urlopen(url, context=ctx) as resp, open(tmp, "wb") as out:
            while chunk := resp.read(1 << 20):
                out.write(chunk)
        tmp.rename(path)
    return path


def _subsample(samples: List[Sample], limit: Optional[int], seed: int) -> List[Sample]:
    if limit is None or limit >= len(samples):
        return samples
    rng = random.Random(seed)
    return sorted(rng.sample(samples, limit), key=lambda s: s.id)


def load_ai4privacy(split: str = "test", limit: Optional[int] = None, seed: int = 13) -> List[Sample]:
    """ai4privacy/pii-masking-400k, English (US + GB locales). Short,
    single-paragraph texts dense with PII - a stress test for recognizer
    coverage across many identifier formats."""
    path = _download("ai4privacy", split)
    samples = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            spans = []
            for m in row["privacy_mask"]:
                cls = AI4PRIVACY_TO_CLASS.get(m["label"])
                if cls is None:
                    raise KeyError(f"unmapped ai4privacy label {m['label']!r}")
                spans.append((m["start"], m["end"], cls))
            samples.append(
                Sample(
                    id=f"ai4p-{row['uid']}",
                    text=row["source_text"],
                    spans=spans,
                    dataset="ai4privacy",
                    meta={"locale": row.get("locale")},
                )
            )
    return _subsample(samples, limit, seed)


def load_gretel(split: str = "test", limit: Optional[int] = None, seed: int = 13) -> List[Sample]:
    """gretelai/synthetic_pii_finance_multilingual, English. Full-length
    financial and legal documents (loan agreements, statements, SWIFT, EDI,
    insurance policies) averaging ~1.4k characters - closest public match to
    the multi-page documents Blacken actually receives.

    Known caveat: annotation is sparse (e.g. "Account Number: 123456789"
    often carries no label), so precision on this set is a lower bound.
    Treat recall as the primary number here."""
    import pandas as pd  # local import: only needed for this dataset

    path = _download("gretel", split)
    df = pd.read_parquet(path)
    samples = []
    for idx, row in df.iterrows():
        spans = []
        for m in json.loads(row["pii_spans"]):
            cls = GRETEL_TO_CLASS.get(m["label"])
            if cls is None:
                raise KeyError(f"unmapped Gretel label {m['label']!r}")
            spans.append((m["start"], m["end"], cls))
        samples.append(
            Sample(
                id=f"gretel-{row.get('__index_level_0__', idx)}",
                text=row["generated_text"],
                spans=spans,
                dataset="gretel",
                meta={"document_type": row["document_type"]},
            )
        )
    return _subsample(samples, limit, seed)


# Hand-annotated gold set markup: [[CLASS|surface text]]
_MARKUP = re.compile(r"\[\[([A-Z_]+)\|(.+?)\]\]", re.DOTALL)
_HEADER = re.compile(r"^=== (?P<id>[\w-]+) \| (?P<domain>[\w-]+) ===$", re.MULTILINE)


def parse_markup(marked: str) -> Tuple[str, List[GoldSpan]]:
    """Turn `Call [[PERSON|Asha Rao]] now` into plain text plus exact
    character offsets, so annotators never hand-count offsets."""
    out: List[str] = []
    spans: List[GoldSpan] = []
    pos = 0
    length = 0
    for m in _MARKUP.finditer(marked):
        before = marked[pos : m.start()]
        out.append(before)
        length += len(before)
        cls, surface = m.group(1), m.group(2)
        spans.append((length, length + len(surface), cls))
        out.append(surface)
        length += len(surface)
        pos = m.end()
    out.append(marked[pos:])
    return "".join(out), spans


def load_blacken_gold(path: Optional[Path] = None) -> List[Sample]:
    """Blacken's own hand-annotated documents: resumes, clinical notes,
    Indian KYC / HR / banking forms, invoices, legal letters, support chats,
    plus PII-free negatives. Every PII occurrence is annotated, so unlike the
    public sets, precision here is meaningful."""
    path = path or DATA_DIR / "blacken_gold.txt"
    raw = path.read_text(encoding="utf-8")
    headers = list(_HEADER.finditer(raw))
    samples = []
    for i, h in enumerate(headers):
        body_end = headers[i + 1].start() if i + 1 < len(headers) else len(raw)
        body = raw[h.end() : body_end].strip("\n")
        text, spans = parse_markup(body)
        samples.append(
            Sample(
                id=h.group("id"),
                text=text,
                spans=spans,
                dataset="blacken_gold",
                meta={"domain": h.group("domain")},
            )
        )
    return samples


def load(name: str, split: str = "test", limit: Optional[int] = None, seed: int = 13) -> List[Sample]:
    if name == "ai4privacy":
        return load_ai4privacy(split, limit, seed)
    if name == "gretel":
        return load_gretel(split, limit, seed)
    if name == "blacken_gold":
        return _subsample(load_blacken_gold(), limit, seed)
    raise ValueError(f"unknown dataset {name!r}")


def iter_core_spans(sample: Sample) -> Iterator[GoldSpan]:
    return (s for s in sample.spans if s[2] != IGNORE)
