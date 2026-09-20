# Single-stage build. The memory-slimmed spaCy model is committed to the repo
# under models/ and copied in as-is - it is deliberately NOT built here.
#
# This image used to have a "model-prep" stage that downloaded the full
# en_core_web_lg (~560MB) and ran nlp.vocab.prune_vectors(20000) at build
# time. That step peaks at ~1.28GB RSS (measured): prune_vectors holds the
# original 342,918x300 vector table (411MB), the 20k-row `keep` copy and the
# 322,918-row `toss` copy all live at once, on top of materializing a Lexeme
# for every vector key. A free-tier builder has less RAM than that, so the
# step got OOM-killed and the build failed with "Failed to build an image"
# after ~5 minutes - with no obvious error, because an OOM kill is a bare
# SIGKILL, not a Python traceback.
#
# The prune is a pure, deterministic, one-off transformation, so it belongs in
# version control, not in every build. To regenerate it (needs ~1.5GB RAM,
# takes ~50s), run:  python scripts/build_pruned_model.py
FROM python:3.11-slim

WORKDIR /app

# Install Tesseract OCR and build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    libtesseract-dev \
    libleptonica-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (layer cached unless requirements change)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application source, including models/en_core_web_lg_pruned. See
# .dockerignore: everything under models/ is excluded except that one
# directory, so stray local model dirs don't bloat the image.
COPY . .

# Must point at the model actually present in this image, or the app fails
# to start ("Can't find model ..."). Overridable at deploy time via a
# platform env var of the same name - e.g. set it to en_core_web_lg if you
# switch to a host with >= 1.5GB RAM and install the full model.
#
# The committed model declares spacy_version ">=3.8.0,<3.9.0" in its
# meta.json, which is why requirements.txt pins spacy to that same range -
# an unpinned spacy would eventually resolve to 4.x and refuse to load it.
ENV SPACY_MODEL=/app/models/en_core_web_lg_pruned

EXPOSE 8000

# Shell form (not exec-array form) so ${PORT} actually expands - Railway
# (and most PaaS hosts) assign a container a port dynamically via this env
# var rather than letting it bind a fixed one.
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}
