#!/usr/bin/env bash
# Deploy the backend to a Hugging Face Space (free tier: 2 vCPU / 16GB RAM).
#
# Why this exists: the app needs ~430-480MB resident, which does not reliably
# fit a 512MB free-tier host. HF Spaces' free CPU tier has 16GB and needs no
# credit card, so the pruned en_core_web_lg model can be kept as-is rather
# than downgrading detection accuracy to en_core_web_sm.
#
# A Space is its own git repo with the Dockerfile at ITS root, while this
# repo keeps the backend in backend/. Rather than restructure, this script
# mirrors backend/ into a checkout of the Space repo and pushes it, so this
# repo stays the single source of truth.
#
# Prerequisites:
#   1. A free account at https://huggingface.co/join
#   2. Create a Space: https://huggingface.co/new-space
#        SDK = Docker, template = Blank, hardware = CPU basic (free)
#   3. An access token with WRITE scope:
#        https://huggingface.co/settings/tokens
#   4. git-lfs installed (brew install git-lfs)
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

command -v git-lfs >/dev/null 2>&1 || {
  echo "git-lfs is required (the model's vocab/vectors file is ~23MB, and HF" >&2
  echo "rejects non-LFS files over 10MB). Install it with: brew install git-lfs" >&2
  exit 1
}

echo "==> Cloning Space $HF_USER/$HF_SPACE"
git clone "https://$HF_USER:$HF_TOKEN@huggingface.co/spaces/$HF_USER/$HF_SPACE" "$WORK/space"
cd "$WORK/space"
git lfs install --local

# Mirror backend/ into the Space, minus anything that must never ship.
echo "==> Mirroring backend/ into the Space"
rsync -a --delete \
  --exclude '.git/' \
  --exclude 'venv/' \
  --exclude '__pycache__/' \
  --exclude '.pytest_cache/' \
  --exclude 'uploads/' \
  --exclude 'data/' \
  --exclude '.env' \
  --exclude 'README.md' \
  "$BACKEND/" ./

# HF serves whatever port app_port names; the Dockerfile's CMD already
# defaults to 8000 when $PORT is unset, so they agree.
cat > README.md <<'EOF'
---
title: Blacken
emoji: 🖤
colorFrom: gray
colorTo: gray
sdk: docker
app_port: 8000
pinned: false
---

# Blacken API

Document PII detection and redaction. See the repository for full docs.

Health check: `/api/health` · API docs: `/api/docs`
EOF

# HF rejects plain files over 10MB. Only the vector table exceeds that, but
# track the whole model dir's binaries so a future re-prune can't trip it.
echo "==> Configuring git-lfs"
git lfs track "models/**/vectors" "models/**/model" "models/**/*.bin" >/dev/null
git add .gitattributes

git add -A
if git diff --cached --quiet; then
  echo "==> No changes to deploy."
  exit 0
fi
git -c user.email="deploy@local" -c user.name="deploy" \
  commit -q -m "Deploy backend from $(cd "$REPO_ROOT" && git rev-parse --short HEAD)"

echo "==> Pushing (this uploads ~54MB of model on the first run)"
git push

cat <<EOF

==> Done. Space: https://huggingface.co/spaces/$HF_USER/$HF_SPACE
    API base:    https://$HF_USER-$HF_SPACE.hf.space

Two settings still to make, in the Space's Settings tab:

  1. Variables and secrets -> New variable:
       CORS_ORIGINS = https://<your-app>.vercel.app
     Without this the browser blocks every request; the code default is
     localhost-only.

  2. In Vercel, set:
       VITE_API_URL = https://$HF_USER-$HF_SPACE.hf.space/api
     then redeploy the frontend.

First build takes a few minutes. Watch it in the Space's "Logs" tab.
EOF
