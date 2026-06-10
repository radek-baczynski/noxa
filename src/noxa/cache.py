from __future__ import annotations

import asyncio
import json
import sqlite3
import struct
import time
from pathlib import Path
from typing import Any

import sqlite_vec

from noxa.config import Settings


def _serialize(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False).encode("utf-8")


def _deserialize(raw: bytes) -> Any:
    return json.loads(raw.decode("utf-8"))


def pack_f32(vector: list[float]) -> bytes:
    return struct.pack(f"{len(vector)}f", *vector)


class CacheStore:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._path = Path(settings.sqlite_path)
        self._conn: sqlite3.Connection | None = None

    def _connect(self) -> sqlite3.Connection:
        if self._conn is None:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(str(self._path), check_same_thread=False)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS cache (
                    key TEXT PRIMARY KEY,
                    value BLOB NOT NULL,
                    expires_at REAL NOT NULL
                )
                """
            )
            conn.enable_load_extension(True)
            sqlite_vec.load(conn)
            conn.enable_load_extension(False)
            dims = self.settings.embedding_dimensions
            conn.execute(
                f"""
                CREATE VIRTUAL TABLE IF NOT EXISTS vec_embeddings USING vec0(
                    embedding float[{dims}]
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS embedding_keys (
                    rowid INTEGER PRIMARY KEY,
                    cache_key TEXT UNIQUE NOT NULL,
                    created_at REAL NOT NULL
                )
                """
            )
            conn.commit()
            self._conn = conn
            self._maybe_migrate_runtime_cache(conn)
        return self._conn

    def _maybe_migrate_runtime_cache(self, conn: sqlite3.Connection) -> None:
        """Wipe stale embedding vectors after elastic-runtime cache key change."""
        marker_key = "__noxa_runtime_cache_version__"
        row = conn.execute(
            "SELECT value FROM cache WHERE key = ?", (marker_key,)
        ).fetchone()
        if row is not None:
            try:
                if _deserialize(row["value"]) == "elastic-runtime-v1":
                    return
            except (json.JSONDecodeError, UnicodeDecodeError):
                pass
        conn.execute("DELETE FROM vec_embeddings")
        conn.execute("DELETE FROM embedding_keys")
        expires_at = time.time() + 10 * 365 * 24 * 3600
        conn.execute(
            "INSERT OR REPLACE INTO cache (key, value, expires_at) VALUES (?, ?, ?)",
            (marker_key, _serialize("elastic-runtime-v1"), expires_at),
        )
        conn.commit()

    async def get(self, key: str) -> Any | None:
        return await asyncio.to_thread(self._get_sync, key)

    def _get_sync(self, key: str) -> Any | None:
        conn = self._connect()
        row = conn.execute(
            "SELECT value, expires_at FROM cache WHERE key = ?", (key,)
        ).fetchone()
        if row is None:
            return None
        if row["expires_at"] < time.time():
            conn.execute("DELETE FROM cache WHERE key = ?", (key,))
            conn.commit()
            return None
        return _deserialize(row["value"])

    async def set(self, key: str, value: Any, ttl_seconds: int) -> None:
        await asyncio.to_thread(self._set_sync, key, value, ttl_seconds)

    def _set_sync(self, key: str, value: Any, ttl_seconds: int) -> None:
        conn = self._connect()
        expires_at = time.time() + ttl_seconds
        conn.execute(
            "INSERT OR REPLACE INTO cache (key, value, expires_at) VALUES (?, ?, ?)",
            (key, _serialize(value), expires_at),
        )
        conn.commit()

    async def get_embeddings(self, keys: list[str]) -> dict[str, list[float]]:
        return await asyncio.to_thread(self._get_embeddings_sync, keys)

    def _get_embeddings_sync(self, keys: list[str]) -> dict[str, list[float]]:
        if not keys:
            return {}
        conn = self._connect()
        now = time.time()
        ttl = self.settings.embedding_ttl_seconds
        placeholders = ",".join("?" * len(keys))
        rows = conn.execute(
            f"""
            SELECT ek.cache_key, ve.embedding, ek.created_at
            FROM embedding_keys ek
            JOIN vec_embeddings ve ON ve.rowid = ek.rowid
            WHERE ek.cache_key IN ({placeholders})
            """,
            keys,
        ).fetchall()
        out: dict[str, list[float]] = {}
        for row in rows:
            if now - row["created_at"] > ttl:
                continue
            blob = row["embedding"]
            n = len(blob) // 4
            out[row["cache_key"]] = list(struct.unpack(f"{n}f", blob))
        return out

    async def set_embeddings(self, items: dict[str, list[float]]) -> None:
        await asyncio.to_thread(self._set_embeddings_sync, items)

    def _set_embeddings_sync(self, items: dict[str, list[float]]) -> None:
        if not items:
            return
        conn = self._connect()
        now = time.time()
        for key, vector in items.items():
            existing = conn.execute(
                "SELECT rowid FROM embedding_keys WHERE cache_key = ?", (key,)
            ).fetchone()
            if existing:
                conn.execute(
                    "DELETE FROM vec_embeddings WHERE rowid = ?", (existing["rowid"],)
                )
                conn.execute(
                    "DELETE FROM embedding_keys WHERE rowid = ?", (existing["rowid"],)
                )
            cur = conn.execute(
                "INSERT INTO vec_embeddings(embedding) VALUES (?)",
                (pack_f32(vector),),
            )
            rowid = cur.lastrowid
            conn.execute(
                "INSERT INTO embedding_keys(rowid, cache_key, created_at) VALUES (?, ?, ?)",
                (rowid, key, now),
            )
        conn.commit()

    async def knn_embeddings(
        self, query_vector: list[float], keys: list[str], k: int
    ) -> list[tuple[str, float]]:
        return await asyncio.to_thread(self._knn_embeddings_sync, query_vector, keys, k)

    def _knn_embeddings_sync(
        self, query_vector: list[float], keys: list[str], k: int
    ) -> list[tuple[str, float]]:
        if not keys:
            return []
        cached = self._get_embeddings_sync(keys)
        if not cached:
            return []
        q = query_vector
        scored: list[tuple[str, float]] = []
        for key, vec in cached.items():
            dot = sum(a * b for a, b in zip(q, vec, strict=True))
            scored.append((key, dot))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]

    async def purge_fetch_cache(self) -> int:
        """Remove cached page payloads (poisoned URL mappings, stale extracts)."""
        return await asyncio.to_thread(self._purge_fetch_cache_sync)

    def _purge_fetch_cache_sync(self) -> int:
        conn = self._connect()
        rows = conn.execute("SELECT key, value FROM cache").fetchall()
        deleted = 0
        for row in rows:
            try:
                val = _deserialize(row["value"])
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
            if isinstance(val, dict) and "text" in val and "url" in val:
                conn.execute("DELETE FROM cache WHERE key = ?", (row["key"],))
                deleted += 1
        conn.commit()
        return deleted

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None
