"""Bracket-style redaction for the plain-text preview shown in the UI.

The actual output documents (PDF/DOCX/image) are redacted at the
content/pixel level by the dedicated redactors in this package, not by
string substitution - this is only for the `redacted_text` preview field.
"""

from typing import List

from app.redaction.matching import find_matches, selected_texts

REDACTION_CHAR = "█"


def redact_txt_file(original_path: str, entities: List[dict], output_path: str) -> int:
    """Plain-text uploads were accepted and analysed, but /redact rejected
    them with "not supported" - so a .txt could be scanned and never
    redacted. Matched spans are replaced character-for-character so line
    lengths and layout survive."""
    for encoding in ("utf-8", "latin-1"):
        try:
            with open(original_path, "r", encoding=encoding) as f:
                text = f.read()
            break
        except UnicodeDecodeError:
            continue
    chars = list(text)
    matches = find_matches(text, selected_texts(entities))
    for start, end in matches:
        for i in range(start, end):
            if not chars[i].isspace():
                chars[i] = REDACTION_CHAR
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("".join(chars))
    return len(matches)


def redact_text(text: str, entities: List[dict]) -> str:
    selected = [e for e in entities if e.get("selected", True)]
    if not selected:
        return text

    selected = sorted(selected, key=lambda e: e["start"], reverse=True)
    redacted = text
    for entity in selected:
        redacted = redacted[: entity["start"]] + "[REDACTED]" + redacted[entity["end"] :]
    return redacted
