"""Encryption at rest for stored documents and their extracted PII.

Blacken keeps every upload for up to RETENTION_HOURS, and its database holds
the most sensitive data in the system: the extracted plain text and the list
of PII values found in it. If DATA_ENCRYPTION_KEY is set (a Fernet key -
generate with `python -c "from cryptography.fernet import Fernet;
print(Fernet.generate_key().decode())"`), then:

* uploaded originals and redacted outputs are written to disk encrypted;
* the extracted text, entities and redacted text columns are encrypted in
  SQLite.

Processing decrypts into a private temp file for the duration of one
request and deletes it afterwards. Rows/files written before a key was
configured are still readable (plaintext is detected and passed through), so
turning encryption on needs no migration. Without a key everything behaves
exactly as before.

Fernet = AES-128-CBC + HMAC-SHA256 with a timestamp; the key must be stored
as a platform secret, never in the repo.
"""

import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional

_FERNET_PREFIX = b"gAAAAA"  # every Fernet token starts with version byte 0x80


class Vault:
    def __init__(self, key: Optional[str]):
        self._fernet = None
        if key:
            from cryptography.fernet import Fernet

            self._fernet = Fernet(key.encode() if isinstance(key, str) else key)

    @property
    def enabled(self) -> bool:
        return self._fernet is not None

    # ---- bytes / text ----
    def encrypt_bytes(self, data: bytes) -> bytes:
        return self._fernet.encrypt(data) if self._fernet else data

    def decrypt_bytes(self, data: bytes) -> bytes:
        if self._fernet and data.startswith(_FERNET_PREFIX):
            return self._fernet.decrypt(data)
        return data

    def encrypt_text(self, text: Optional[str]) -> Optional[str]:
        if text is None or not self._fernet:
            return text
        return self._fernet.encrypt(text.encode("utf-8")).decode("ascii")

    def decrypt_text(self, value: Optional[str]) -> Optional[str]:
        if value is None or not self._fernet or not value.startswith(_FERNET_PREFIX.decode()):
            return value
        return self._fernet.decrypt(value.encode("ascii")).decode("utf-8")

    # ---- files ----
    def write_file(self, path: Path, data: bytes) -> None:
        tmp = Path(str(path) + ".part")
        tmp.write_bytes(self.encrypt_bytes(data))
        os.replace(tmp, path)

    def read_file(self, path: Path) -> bytes:
        return self.decrypt_bytes(Path(path).read_bytes())

    def encrypt_file_in_place(self, path: Path) -> None:
        if self._fernet:
            self.write_file(Path(path), Path(path).read_bytes())

    @contextmanager
    def plaintext_path(self, path: str, suffix: str = "") -> Iterator[str]:
        """Yield a filesystem path holding the decrypted file (the file itself
        when encryption is off), deleting any temp copy afterwards."""
        if not self._fernet:
            yield path
            return
        fd, tmp = tempfile.mkstemp(suffix=suffix or Path(path).suffix, prefix="blacken-")
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(self.read_file(Path(path)))
            yield tmp
        finally:
            try:
                os.remove(tmp)
            except OSError:
                pass
