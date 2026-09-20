#!/usr/bin/env bash
# Deploy the backend to a Hugging Face Space (free tier: 2 vCPU / 16GB RAM,
# no credit card).
#
# Why a separate path from the Railway image: with 16GB there is no memory
# pressure, so the Space downloads the FULL en_core_web_lg at build time
# rather than shipping the pruned model. Two consequences, both good:
#   - The Space repo carries no large files, so git-lfs is not needed (HF
#     rejects non-LFS files over 10MB, and the pruned model's vector table
#     is ~23MB).
#   - Detection runs on unpruned en_core_web_lg. Accuracy is identical to
#     the pruned model (same NER weights, F=0.855), so this is for
#     simplicity, not accuracy.
#
# A Space is its own git repo expecting the Dockerfile at ITS root, while
# this repo keeps the backend in backend/. This mirrors backend/ into a
# checkout of the Space and pushes, so this repo stays the source of truth.
#
# Prerequisites:
#   1. Free account:  https://huggingface.co/join
#   2. Create a Space: https://huggingface.co/new-space
#        SDK = Docker, template = Blank, hardware = CPU basic (free)
#   3. WRITE-scoped token: https://huggingface.co/settings/tokens
#
# Usage:
#   HF_USER=your-username HF_SPACE=blacken HF_TOKEN=hf_xxx \
#     ./scripts/deploy_hf_space.sh

set -euo pipefail

: "${HF_USER:?set HF_USER to your Hugging Face username}"
: "${HF_SPACE:?set HF_SPACE to your Space name}"
: "${HF_TOKEN:?set HF_TOKEN to a WRITE-scoped token from https://huggingface.co/settings/tokens}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BACKEND="$REPO_ROOT/backend"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

echo "==> Cloning Space $HF_USER/$HF_SPACE"
git clone "https://$HF_USER:$HF_TOKEN@huggingface.co/spaces/$HF_USER/$HF_SPACE" "$WORK/space"
cd "$WORK/space"

# Mirror backend/, minus anything that must not ship. models/ is excluded on
# purpose - see the header: the Space builds its own full model, and keeping
# it out is what avoids needing git-lfs.
echo "==> Mirroring backend/ (excluding models/)"
rsync -a --delete \
  --exclude '.git/' --exclude '.gitattributes' \
  --exclude 'venv/' --exclude '__pycache__/' --exclude '.pytest_cache/' \
  --exclude 'uploads/' --exclude 'data/' --exclude 'models/' \
  --exclude '.DS_Store' --exclude 'railway.json' \
  --exclude '.env' --exclude 'README.md' --exclude 'Dockerfile' \
  "$BACKEND/" ./

# 7860 is HF's default Space port. Hardcoding it on both sides (CMD and
# app_port below) means there is no PORT env var to disagree about.
cat > Dockerfile <<'EOF'
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr libtesseract-dev libleptonica-dev gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Unlike the Railway image, this host has the RAM for the full model, so
# fetch it normally instead of shipping a pruned one. ~560MB download,
# ~800MB resident - comfortable inside 16GB.
RUN python -m spacy download en_core_web_lg
ENV SPACY_MODEL=en_core_web_lg

COPY . .

# The app creates these at import time. Spaces may run the container as a
# non-root user, so make them writable either way.
RUN mkdir -p /app/uploads /app/data && chmod 777 /app/uploads /app/data

EXPOSE 7860
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
EOF

cat > README.md <<'EOF'
---
title: Blacken
emoji: 🖤
colorFrom: gray
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
---

# Blacken API

Document PII detection and redaction.

- Health: `/api/health`
- API docs: `/api/docs`
EOF

git add -A
if git diff --cached --quiet; then
  echo "==> No changes to deploy."
  exit 0
fi
git -c user.email="deploy@local" -c user.name="deploy" \
  commit -q -m "Deploy backend from $(cd "$REPO_ROOT" && git rev-parse --short HEAD)"

echo "==> Pushing"
git push

cat <<EOF

==> Done. Space: https://huggingface.co/spaces/$HF_USER/$HF_SPACE
    API base:    https://$HF_USER-$HF_SPACE.hf.space/api

Still to do:

  1. Space -> Settings -> Variables and secrets -> New variable:
       CORS_ORIGINS = https://<your-app>.vercel.app
     Without it the browser blocks every request.

  2. Vercel -> Environment Variables:
       VITE_API_URL = https://$HF_USER-$HF_SPACE.hf.space/api
     Then REDEPLOY the frontend (Vite bakes env vars in at build time).

First build takes several minutes (it downloads the model). Watch the
Space's "Logs" tab; startup is done when you see "Blacken API ready."
EOF
