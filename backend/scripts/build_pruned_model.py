"""Regenerate models/en_core_web_lg_pruned.

This is the memory-slimmed spaCy model the deployed image runs on. It is
committed to the repo rather than built in the Dockerfile, because the prune
peaks at ~1.28GB RSS and gets OOM-killed on a free-tier build machine (see the
comment at the top of the Dockerfile).

Full en_core_web_lg loads to ~700MB RSS - its 342,918-entry word-vector table
dominates. Those vectors are only an auxiliary lookup for the NER model, whose
own trained weights are a small fraction of that, so spaCy's own supported
technique (Vocab.prune_vectors) collapses rare vectors onto their nearest
remaining neighbour. Measured here: 700MB -> 364MB model RSS, with no
meaningful NER difference on real documents.

Run this only when you want to change the vector count or move to a new spaCy
release. It needs ~1.5GB of free RAM and takes about a minute.

    python scripts/build_pruned_model.py
"""

import shutil
import subprocess
import sys
import time
from pathlib import Path

SOURCE_MODEL = "en_core_web_lg"
KEEP_VECTORS = 20_000
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "models" / "en_core_web_lg_pruned"


def main() -> int:
    try:
        import spacy
    except ImportError:
        print("spaCy is not installed. Install requirements.txt first.", file=sys.stderr)
        return 1

    try:
        nlp = spacy.load(SOURCE_MODEL)
    except OSError:
        print(f"{SOURCE_MODEL} is not installed. Fetching it...")
        subprocess.check_call([sys.executable, "-m", "spacy", "download", SOURCE_MODEL])
        nlp = spacy.load(SOURCE_MODEL)

    before = nlp.vocab.vectors.shape
    print(f"Loaded {SOURCE_MODEL}: {before[0]:,} vectors x {before[1]}")

    print(f"Pruning to {KEEP_VECTORS:,} vectors (needs ~1.5GB RAM)...")
    start = time.time()
    nlp.vocab.prune_vectors(KEEP_VECTORS)
    print(f"Pruned in {time.time() - start:.0f}s -> {nlp.vocab.vectors.shape[0]:,} vectors")

    # to_disk() only creates the leaf directory (a plain mkdir(), not
    # mkdir(parents=True)), so its parent must already exist.
    OUTPUT_DIR.parent.mkdir(parents=True, exist_ok=True)
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    nlp.to_disk(OUTPUT_DIR)

    size_mb = sum(f.stat().st_size for f in OUTPUT_DIR.rglob("*") if f.is_file()) / 1024**2
    print(f"Wrote {OUTPUT_DIR} ({size_mb:.0f}MB)")

    # Load it back: a model that writes cleanly but won't reload is the one
    # failure mode that would otherwise only show up on the deployed host.
    reloaded = spacy.load(OUTPUT_DIR)
    ents = reloaded("Michael Andersen lives in San Francisco.").ents
    print(f"Verified reload. Sample NER: {[(e.text, e.label_) for e in ents]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
