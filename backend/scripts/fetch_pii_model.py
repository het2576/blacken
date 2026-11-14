"""Download the optional GLiNER PII transformer used by
app/detection/gliner_onnx.py.

    python scripts/fetch_pii_model.py                  # -> models/gliner-pii-edge
    python scripts/fetch_pii_model.py --dest /app/models/gliner-pii-edge

Pinned to an exact Hugging Face revision and verified by SHA-256, so a build
is reproducible and can never silently pick up a changed upstream model.
knowledgator/gliner-pii-edge-v1.0 is Apache-2.0; the int8 ONNX export is
~46 MB and runs on onnxruntime alone (no PyTorch).

Then set PII_MODEL_DIR to the destination directory.
"""

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

REPO = "knowledgator/gliner-pii-edge-v1.0"
REVISION = "9b7f39b0a2da971a5beea78d35f1539d4009c891"
FILES = {
    "onnx/model_quint8.onnx": "988acb03456b26e2d9f2521016d820310c2ed64deb4a846297d3289f0c2eb7e4",
    "tokenizer.json": "84b3a9b18f04a0ccd03b72d9f871b7e0bec40fd7021ef50bc30a7c3693c11205",
    "gliner_config.json": "77e6b57335c4bfd461e9041682196dd6c373a0b09bbd9269ef9e95b807915340",
}
DEFAULT_DEST = Path(__file__).resolve().parent.parent / "models" / "gliner-pii-edge"


def _ssl_context():
    try:
        import ssl

        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return None


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", type=Path, default=DEFAULT_DEST)
    args = ap.parse_args(argv)
    args.dest.mkdir(parents=True, exist_ok=True)
    ctx = _ssl_context()

    for remote, expected in FILES.items():
        target = args.dest / Path(remote).name
        if target.exists() and sha256(target) == expected:
            print(f"ok      {target.name}")
            continue
        url = f"https://huggingface.co/{REPO}/resolve/{REVISION}/{remote}"
        tmp = target.with_suffix(target.suffix + ".part")
        print(f"fetch   {target.name}")
        with urllib.request.urlopen(url, context=ctx) as resp, open(tmp, "wb") as out:
            while chunk := resp.read(1 << 20):
                out.write(chunk)
        actual = sha256(tmp)
        if actual != expected:
            tmp.unlink(missing_ok=True)
            print(f"SHA-256 mismatch for {remote}: {actual} != {expected}", file=sys.stderr)
            return 1
        tmp.rename(target)
    print(f"model ready in {args.dest} - set PII_MODEL_DIR={args.dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
