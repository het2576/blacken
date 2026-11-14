"""SQLite-backed document store.

Documents/entities survive a server restart, and `purge_expired` lets a
background task actually delete old PII instead of letting it accumulate.

Uses plain stdlib sqlite3 (not an async driver) wrapped in
`asyncio.to_thread`, which is plenty for this access pattern and avoids
pulling in another dependency.

Three tables:
  documents   - one row per upload. The PII-bearing columns (extracted text,
                entities, redacted text/entities) are encrypted at rest when a
                Vault key is configured (see app/vault.py).
  feedback    - reviewer decisions on detected entities: type, confidence
                bucket and outcome only - never the entity text. This is what
                measures real-world precision per entity type.
  preferences - per-owner custom terms ("always redact" / "never flag"),
                encrypted, since a list of client names is itself PII.
"""

import asyncio
import json
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from app.vault import Vault

_JSON_COLUMNS = {"entities", "redacted_entities"}
_ENCRYPTED_COLUMNS = {"extracted_text", "entities", "redacted_text", "redacted_entities"}


class DocumentStore:
    def __init__(self, db_path: Path, vault: Optional[Vault] = None):
        self.db_path = db_path
        self.vault = vault or Vault(None)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        conn = self._connect()
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    owner_id TEXT,
                    filename TEXT NOT NULL,
                    upload_time TEXT NOT NULL,
                    original_path TEXT,
                    extracted_text TEXT,
                    entities TEXT,
                    status TEXT,
                    redacted_path TEXT,
                    redacted_filename TEXT,
                    redacted_count INTEGER,
                    redacted_entities TEXT,
                    redacted_text TEXT,
                    redaction_time TEXT,
                    created_at REAL NOT NULL
                )
                """
            )
            # Migrate databases created before per-owner scoping existed. Rows
            # from that era have a NULL owner_id and therefore match no client,
            # which is exactly what we want: previously shared documents stop
            # appearing in everyone's history and age out via retention.
            columns = {row["name"] for row in conn.execute("PRAGMA table_info(documents)")}
            if "owner_id" not in columns:
                conn.execute("ALTER TABLE documents ADD COLUMN owner_id TEXT")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_documents_owner ON documents(owner_id)")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS feedback (
                    created_at REAL NOT NULL,
                    entity_type TEXT NOT NULL,
                    confidence_bucket REAL,
                    outcome TEXT NOT NULL,      -- accepted | rejected | added
                    was_preselected INTEGER
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS preferences (
                    owner_id TEXT PRIMARY KEY,
                    always_redact TEXT,
                    never_flag TEXT,
                    updated_at REAL NOT NULL
                )
                """
            )
            conn.commit()
        finally:
            conn.close()

    # ---- (de)serialisation with encryption ----
    def _encode(self, key: str, value: Any) -> Any:
        if key in _JSON_COLUMNS:
            value = json.dumps(value or [])
        if key in _ENCRYPTED_COLUMNS:
            value = self.vault.encrypt_text(value)
        return value

    def _row_to_dict(self, row: sqlite3.Row) -> dict:
        doc = dict(row)
        for key in _ENCRYPTED_COLUMNS & doc.keys():
            doc[key] = self.vault.decrypt_text(doc[key])
        for key in _JSON_COLUMNS & doc.keys():
            doc[key] = json.loads(doc[key]) if doc.get(key) else []
        return doc

    # -- blocking implementations (run via asyncio.to_thread) --

    def _create(self, document_id: str, data: dict) -> None:
        conn = self._connect()
        try:
            conn.execute(
                """
                INSERT INTO documents
                    (id, owner_id, filename, upload_time, original_path, extracted_text,
                     entities, status, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    document_id,
                    data.get("owner_id"),
                    data["filename"],
                    data["upload_time"],
                    data["original_path"],
                    self._encode("extracted_text", data["extracted_text"]),
                    self._encode("entities", data.get("entities", [])),
                    data.get("status", "processed"),
                    time.time(),
                ),
            )
            conn.commit()
        finally:
            conn.close()

    def _get(self, document_id: str, owner_id: Optional[str] = None) -> Optional[dict]:
        conn = self._connect()
        try:
            if owner_id is None:
                row = conn.execute("SELECT * FROM documents WHERE id = ?", (document_id,)).fetchone()
            else:
                row = conn.execute(
                    "SELECT * FROM documents WHERE id = ? AND owner_id = ?",
                    (document_id, owner_id),
                ).fetchone()
            return self._row_to_dict(row) if row else None
        finally:
            conn.close()

    _UPDATABLE = {
        "status", "redacted_path", "redacted_filename", "redacted_count",
        "redacted_entities", "redacted_text", "redaction_time", "entities",
    }

    def _update(self, document_id: str, updates: dict) -> None:
        if not updates:
            return
        unknown = set(updates) - self._UPDATABLE
        if unknown:  # column names are interpolated below - never allow arbitrary keys
            raise ValueError(f"Not updatable: {sorted(unknown)}")
        conn = self._connect()
        try:
            set_clause = [f"{key} = ?" for key in updates]
            values: list[Any] = [self._encode(k, v) for k, v in updates.items()]
            values.append(document_id)
            conn.execute(f"UPDATE documents SET {', '.join(set_clause)} WHERE id = ?", values)
            conn.commit()
        finally:
            conn.close()

    def _list(self, owner_id: str) -> list[dict]:
        conn = self._connect()
        try:
            rows = conn.execute(
                """
                SELECT id, filename, upload_time, status, entities, redacted_count, created_at
                FROM documents
                WHERE owner_id = ?
                ORDER BY created_at DESC
                """,
                (owner_id,),
            ).fetchall()
            return [self._row_to_dict(row) for row in rows]
        finally:
            conn.close()

    def _delete(self, document_id: str, owner_id: Optional[str] = None) -> None:
        conn = self._connect()
        try:
            if owner_id is None:
                conn.execute("DELETE FROM documents WHERE id = ?", (document_id,))
            else:
                conn.execute(
                    "DELETE FROM documents WHERE id = ? AND owner_id = ?", (document_id, owner_id)
                )
            conn.commit()
        finally:
            conn.close()

    def _purge_expired(self, retention_seconds: float) -> list[dict]:
        conn = self._connect()
        try:
            cutoff = time.time() - retention_seconds
            rows = conn.execute(
                "SELECT id, original_path, redacted_path FROM documents WHERE created_at < ?",
                (cutoff,),
            ).fetchall()
            expired = [dict(row) for row in rows]
            conn.execute("DELETE FROM documents WHERE created_at < ?", (cutoff,))
            conn.commit()
            return expired
        finally:
            conn.close()

    # ---- feedback (no PII) ----
    def _record_feedback(self, events: List[dict]) -> None:
        if not events:
            return
        now = time.time()
        conn = self._connect()
        try:
            conn.executemany(
                "INSERT INTO feedback (created_at, entity_type, confidence_bucket, outcome, was_preselected)"
                " VALUES (?, ?, ?, ?, ?)",
                [
                    (now, e["entity_type"], e.get("confidence_bucket"), e["outcome"], e.get("was_preselected"))
                    for e in events
                ],
            )
            conn.commit()
        finally:
            conn.close()

    def _feedback_summary(self, since_seconds: Optional[float]) -> List[dict]:
        conn = self._connect()
        try:
            where, params = "", ()
            if since_seconds:
                where, params = "WHERE created_at >= ?", (time.time() - since_seconds,)
            rows = conn.execute(
                f"""
                SELECT entity_type, confidence_bucket, outcome, was_preselected, COUNT(*) AS n
                FROM feedback {where}
                GROUP BY entity_type, confidence_bucket, outcome, was_preselected
                """,
                params,
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    # ---- preferences ----
    def _get_prefs(self, owner_id: str) -> Dict[str, List[str]]:
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT always_redact, never_flag FROM preferences WHERE owner_id = ?", (owner_id,)
            ).fetchone()
        finally:
            conn.close()
        if not row:
            return {"always_redact": [], "never_flag": []}
        return {
            key: json.loads(self.vault.decrypt_text(row[key]) or "[]")
            for key in ("always_redact", "never_flag")
        }

    def _set_prefs(self, owner_id: str, always_redact: Iterable[str], never_flag: Iterable[str]) -> None:
        conn = self._connect()
        try:
            conn.execute(
                """
                INSERT INTO preferences (owner_id, always_redact, never_flag, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(owner_id) DO UPDATE SET
                    always_redact = excluded.always_redact,
                    never_flag = excluded.never_flag,
                    updated_at = excluded.updated_at
                """,
                (
                    owner_id,
                    self.vault.encrypt_text(json.dumps(list(always_redact))),
                    self.vault.encrypt_text(json.dumps(list(never_flag))),
                    time.time(),
                ),
            )
            conn.commit()
        finally:
            conn.close()

    # -- async facade used by routes --

    async def create(self, document_id: str, data: dict) -> None:
        await asyncio.to_thread(self._create, document_id, data)

    async def get(self, document_id: str, owner_id: Optional[str] = None) -> Optional[dict]:
        return await asyncio.to_thread(self._get, document_id, owner_id)

    async def update(self, document_id: str, **updates) -> None:
        await asyncio.to_thread(self._update, document_id, updates)

    async def list_all(self, owner_id: str) -> list[dict]:
        return await asyncio.to_thread(self._list, owner_id)

    async def delete(self, document_id: str, owner_id: Optional[str] = None) -> None:
        await asyncio.to_thread(self._delete, document_id, owner_id)

    async def purge_expired(self, retention_hours: float) -> list[dict]:
        return await asyncio.to_thread(self._purge_expired, retention_hours * 3600)

    async def record_feedback(self, events: List[dict]) -> None:
        await asyncio.to_thread(self._record_feedback, events)

    async def feedback_summary(self, since_seconds: Optional[float] = None) -> List[dict]:
        return await asyncio.to_thread(self._feedback_summary, since_seconds)

    async def get_prefs(self, owner_id: str) -> Dict[str, List[str]]:
        return await asyncio.to_thread(self._get_prefs, owner_id)

    async def set_prefs(self, owner_id: str, always_redact, never_flag) -> None:
        await asyncio.to_thread(self._set_prefs, owner_id, always_redact, never_flag)
