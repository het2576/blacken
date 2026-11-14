"""Request-level protections: file-content validation, rate limiting, and a
bound on concurrent CPU-heavy jobs.

All in-process and dependency-free, which is right for a single-instance
deployment (Railway free tier). If Blacken ever runs more than one replica,
the rate limiter's state must move to a shared store (Redis) - noted where
it matters.
"""

import asyncio
import io
import time
import zipfile
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from typing import Deque, Dict, Optional, Tuple

from fastapi import HTTPException, Request

# --------------------------------------------------------------------------
# Content validation
# --------------------------------------------------------------------------
# The extension says what the client *claims*; the leading bytes say what the
# file *is*. Without this, a renamed binary reaches PyMuPDF / python-docx /
# Pillow parsers - the classic way untrusted uploads exploit parser bugs - and
# mislabelled files fail deep inside extraction with confusing 500s.
_SIGNATURES: Dict[str, Tuple[bytes, ...]] = {
    ".pdf": (b"%PDF-",),
    ".docx": (b"PK\x03\x04",),
    ".png": (b"\x89PNG\r\n\x1a\n",),
    ".jpg": (b"\xff\xd8\xff",),
    ".jpeg": (b"\xff\xd8\xff",),
    ".tiff": (b"II*\x00", b"MM\x00*"),
    ".bmp": (b"BM",),
}


def validate_file_content(ext: str, content: bytes) -> None:
    """Raise 400 unless `content` really is a file of type `ext`."""
    if ext == ".txt":
        if b"\x00" in content[:8192]:
            raise HTTPException(status_code=400, detail="File is not plain text")
        return
    signatures = _SIGNATURES.get(ext)
    if signatures is None:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext}")
    if not any(content.startswith(sig) for sig in signatures):
        raise HTTPException(
            status_code=400, detail=f"File content does not match its {ext} extension"
        )
    if ext == ".docx":
        # Any zip starts with PK; a real DOCX has the main document part.
        try:
            with zipfile.ZipFile(io.BytesIO(content)) as z:
                if "word/document.xml" not in z.namelist():
                    raise HTTPException(status_code=400, detail="Not a valid Word document")
        except zipfile.BadZipFile as exc:
            raise HTTPException(status_code=400, detail="Corrupt Word document") from exc


# --------------------------------------------------------------------------
# Rate limiting
# --------------------------------------------------------------------------
def client_ip(request: Request) -> str:
    """First hop of X-Forwarded-For when behind the platform proxy (Railway,
    Render and Vercel all set it), else the socket peer."""
    forwarded = request.headers.get("x-forwarded-for", "")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


class SlidingWindowRateLimiter:
    """Per-key sliding-window limiter. In-memory, so per process: correct for
    one instance; use Redis if horizontally scaled."""

    def __init__(self, max_keys: int = 50_000):
        self._hits: Dict[str, Deque[float]] = defaultdict(deque)
        self._max_keys = max_keys

    def check(self, key: str, limit: int, window_s: float) -> Optional[float]:
        """Record a hit; return seconds to wait if over the limit, else None."""
        now = time.monotonic()
        hits = self._hits[key]
        while hits and now - hits[0] > window_s:
            hits.popleft()
        if len(hits) >= limit:
            return max(0.0, window_s - (now - hits[0]))
        hits.append(now)
        if len(self._hits) > self._max_keys:  # bound memory under key churn
            for stale in [k for k, v in self._hits.items() if not v][: self._max_keys // 10]:
                del self._hits[stale]
        return None

    def reset(self) -> None:
        self._hits.clear()


rate_limiter = SlidingWindowRateLimiter()


def enforce_rate_limit(request: Request, bucket: str, limit: int, window_s: float = 60.0) -> None:
    owner = request.headers.get("x-owner-id", "")
    # Keyed on IP *and* owner: a new owner token per request doesn't escape
    # the IP limit, and many users behind one NAT don't share one budget.
    for key in (f"{bucket}:ip:{client_ip(request)}", f"{bucket}:owner:{owner}"):
        if key.endswith(":") or limit <= 0:
            continue
        retry_after = rate_limiter.check(key, limit, window_s)
        if retry_after is not None:
            raise HTTPException(
                status_code=429,
                detail="Too many requests - please slow down",
                headers={"Retry-After": str(int(retry_after) + 1)},
            )


# --------------------------------------------------------------------------
# Bounded concurrency for CPU/memory-heavy work
# --------------------------------------------------------------------------
class JobLimiter:
    """Caps how many extraction/detection/redaction jobs run at once.

    On a 512 MB container the model already uses most of the memory; two
    25 MB scanned PDFs rasterised at 300 DPI concurrently is an OOM kill that
    takes every user's request down with it. Excess requests wait briefly in
    line and then get 503 + Retry-After - a degraded answer instead of a
    crashed process."""

    def __init__(self, max_concurrent: int, queue_timeout_s: float):
        self._sem = asyncio.Semaphore(max_concurrent)
        self.max_concurrent = max_concurrent
        self.queue_timeout_s = queue_timeout_s
        self.active = 0
        self.waiting = 0

    @asynccontextmanager
    async def slot(self):
        self.waiting += 1
        try:
            await asyncio.wait_for(self._sem.acquire(), timeout=self.queue_timeout_s)
        except asyncio.TimeoutError as exc:
            raise HTTPException(
                status_code=503,
                detail="Server is busy processing other documents - retry shortly",
                headers={"Retry-After": "10"},
            ) from exc
        finally:
            self.waiting -= 1
        self.active += 1
        try:
            yield
        finally:
            self.active -= 1
            self._sem.release()
