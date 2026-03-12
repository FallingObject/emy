from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

import faiss
import numpy as np

from .config import EmyConfig
from .llm import LLMClient

logger = logging.getLogger(__name__)


class EmbeddingIndex:
    """FAISS + SQLite index for a single namespace (memory or documents)."""

    def __init__(self, config: EmyConfig, llm: LLMClient, namespace: str = "default"):
        self.config = config
        self.llm = llm
        self.namespace = namespace

        idx_dir = config.indexes_dir
        idx_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = idx_dir / f"{namespace}.faiss"
        self.meta_db_path = idx_dir / f"{namespace}_meta.db"

        self._init_db()
        self._load_index()

    # ── database ──────────────────────────────────────────────────────

    def _init_db(self) -> None:
        self.db = sqlite3.connect(str(self.meta_db_path), check_same_thread=False)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute(
            """
            CREATE TABLE IF NOT EXISTS embeddings (
                faiss_id   INTEGER PRIMARY KEY,
                entry_id   TEXT    NOT NULL,
                text       TEXT    NOT NULL,
                source     TEXT    NOT NULL,
                model      TEXT    NOT NULL,
                created_at TEXT    NOT NULL
            )
            """
        )
        self.db.commit()

    # ── FAISS ─────────────────────────────────────────────────────────

    def _load_index(self) -> None:
        if self.index_path.exists():
            self.index = faiss.read_index(str(self.index_path))
        else:
            self.index = faiss.IndexFlatIP(self.config.embedding_dim)

    def _save_index(self) -> None:
        faiss.write_index(self.index, str(self.index_path))

    # ── public API ────────────────────────────────────────────────────

    def add(
        self,
        texts: list[str],
        entry_ids: list[str],
        source: str = "memory",
    ) -> None:
        if not texts:
            return
        vecs = self.llm.embed(texts)
        faiss.normalize_L2(vecs)
        start = self.index.ntotal
        self.index.add(vecs)
        self._save_index()

        from .utils import now_iso

        ts = now_iso()
        for i, (text, eid) in enumerate(zip(texts, entry_ids)):
            self.db.execute(
                "INSERT INTO embeddings (faiss_id, entry_id, text, source, model, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (start + i, eid, text, source, self.config.embedding_model, ts),
            )
        self.db.commit()

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        if self.index.ntotal == 0:
            return []
        q = self.llm.embed([query])
        faiss.normalize_L2(q)
        k = min(top_k, self.index.ntotal)
        distances, indices = self.index.search(q, k)
        results: list[dict] = []
        for score, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue
            row = self.db.execute(
                "SELECT entry_id, text, source FROM embeddings WHERE faiss_id = ?",
                (int(idx),),
            ).fetchone()
            if row:
                results.append(
                    {
                        "entry_id": row[0],
                        "text": row[1],
                        "source": row[2],
                        "score": float(score),
                    }
                )
        return results

    def get_all_entries(self) -> list[dict]:
        rows = self.db.execute(
            "SELECT entry_id, text, source FROM embeddings ORDER BY faiss_id"
        ).fetchall()
        return [{"entry_id": r[0], "text": r[1], "source": r[2]} for r in rows]

    def rebuild(
        self,
        texts: list[str],
        entry_ids: list[str],
        source: str = "memory",
    ) -> None:
        """Drop everything and re-index."""
        self.index = faiss.IndexFlatIP(self.config.embedding_dim)
        self.db.execute("DELETE FROM embeddings")
        self.db.commit()
        if self.index_path.exists():
            self.index_path.unlink()
        if texts:
            self.add(texts, entry_ids, source)

    @property
    def total(self) -> int:
        return self.index.ntotal
