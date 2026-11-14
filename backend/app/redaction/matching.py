"""Locating entity text inside rendered document text, shared by every
redactor (PDF, DOCX, image, plain text).

Two properties matter:

* Token boundaries. A plain substring search for "52" also hits the "52" in
  "1,52,400" or "Ward 352", and the old redactors blacked all of those out -
  measured at 17% of non-PII words destroyed on PDFs. A match must now start
  and end on a token boundary (not flanked by a letter or digit).
* Whitespace tolerance. Extraction and rendering disagree about spacing
  (PDF word reconstruction collapses runs of spaces, DOCX keeps them), so any
  whitespace run in the entity matches any whitespace run in the document.

Every boundary-respecting occurrence is redacted, not only the one the
detector pointed at: if a name appears three times, all three are blacked
out, which is what a reviewer expects from "redact this entity".
"""

import re
from functools import lru_cache
from typing import Iterable, List, Tuple

_WS = re.compile(r"\s+")


@lru_cache(maxsize=4096)
def _pattern(needle: str) -> "re.Pattern[str] | None":
    parts = [re.escape(p) for p in _WS.split(needle.strip()) if p]
    if not parts:
        return None
    body = r"\s+".join(parts)
    # Digits joined by "," or "." are one number ("1,52,400", "3.52"), so a
    # digit-separator neighbour is not a boundary either.
    return re.compile(
        rf"(?<![A-Za-z0-9])(?<!\d[,.]){body}(?![A-Za-z0-9])(?![,.]\d)", re.IGNORECASE
    )


def find_matches(haystack: str, needles: Iterable[str]) -> List[Tuple[int, int]]:
    """All (start, end) spans in `haystack` matching any needle, merged so
    overlapping matches from different entities don't double-count."""
    spans: List[Tuple[int, int]] = []
    for needle in needles:
        pattern = _pattern(needle)
        if pattern is None:
            continue
        spans.extend((m.start(), m.end()) for m in pattern.finditer(haystack))
    spans.sort()
    merged: List[Tuple[int, int]] = []
    for start, end in spans:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def selected_texts(entities: Iterable[dict]) -> List[str]:
    """Unique, non-empty texts of entities the user left selected."""
    seen = set()
    out = []
    for e in entities:
        text = (e.get("text") or "").strip()
        if e.get("selected", True) and text and text.lower() not in seen:
            seen.add(text.lower())
            out.append(text)
    return out
