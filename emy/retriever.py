from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

from .config import EmyConfig
from .embeddings import EmbeddingIndex
from .llm import LLMClient
from .types import RetrievalChunk
from .utils import read_file_text, sliding_chunks
from .vision import image_to_base64, render_pdf_pages

logger = logging.getLogger(__name__)



class HybridRetriever:
    """Dense (FAISS) + lexical (BM25) retrieval with RRF merging."""

    def __init__(self, config: EmyConfig, llm: LLMClient, embedding_index: EmbeddingIndex):
        self.config = config
        self.llm = llm
        self.embedding_index = embedding_index
        self.chunks: list[dict] = []
        self.bm25: BM25Okapi | None = None
        self._load_from_db()

    def _load_from_db(self) -> None:
        rows = self.embedding_index.get_all_entries()
        self.chunks = [
            {"chunk_id": r["entry_id"], "text": r["text"], "source": r["source"]}
            for r in rows
        ]
        if self.chunks:
            tokenized = [c["text"].lower().split() for c in self.chunks]
            self.bm25 = BM25Okapi(tokenized)
        else:
            self.bm25 = None

    def ingest(self, corpus_dir: str) -> int:
        corpus_path = Path(corpus_dir)
        if not corpus_path.exists():
            raise FileNotFoundError(f"Corpus directory not found: {corpus_dir}")

        chunks: list[dict] = []
        for path in sorted(corpus_path.rglob("*")):
            if path.is_dir():
                continue
            text = read_file_text(path)

            # Vision fallback for weak PDF extraction (future: qwen2.5-vl)
            if (
                self.config.vision_fallback_enabled
                and path.suffix.lower() == ".pdf"
            ):
                vision_text = self._vision_extract_pdf(path)
                if vision_text.strip():
                    text = vision_text

            if not text.strip():
                continue
            for i, chunk_text in enumerate(
                sliding_chunks(text, self.config.chunk_size, self.config.chunk_overlap)
            ):
                chunk_id = f"chk_{len(chunks):05d}"
                chunks.append(
                    {
                        "chunk_id": chunk_id,
                        "source": str(path.relative_to(corpus_path)),
                        "text": chunk_text,
                    }
                )

        if not chunks:
            return 0

        texts = [c["text"] for c in chunks]
        ids = [c["chunk_id"] for c in chunks]
        sources = [c["source"] for c in chunks]
        # Rebuild dense index with per-chunk source info
        self.embedding_index.rebuild(texts, ids, source="documents")
        # Patch source column to actual file paths
        for fid, src in enumerate(sources):
            self.embedding_index.db.execute(
                "UPDATE embeddings SET source = ? WHERE faiss_id = ?", (src, fid)
            )
        self.embedding_index.db.commit()

        # Rebuild BM25
        self.chunks = chunks
        tokenized = [c["text"].lower().split() for c in chunks]
        self.bm25 = BM25Okapi(tokenized)

        logger.info("Ingested %d chunks from %s", len(chunks), corpus_dir)
        return len(chunks)

    def _vision_extract_pdf(self, path: Path) -> str:
        """Render PDF pages to images and extract text via the vision model."""
        images = render_pdf_pages(path)
        if not images:
            return ""

        # Process in batches of 4 pages to stay within context limits
        batch_size = 4
        parts: list[str] = []
        for i in range(0, len(images), batch_size):
            batch = images[i : i + batch_size]
            b64_batch = [image_to_base64(img) for img in batch]
            extracted = self.llm.vision_extract(b64_batch)
            if extracted.strip():
                parts.append(extracted)

        result = "\n\n".join(parts)
        if result:
            logger.info(
                "Vision fallback extracted %d chars from %s (%d pages)",
                len(result), path.name, len(images),
            )
        return result

    def search(self, query: str, top_k: int | None = None) -> list[RetrievalChunk]:
        top_k = top_k or self.config.max_retrieval_hits
        fetch_k = top_k * 2

        # Dense retrieval
        dense = self.embedding_index.search(query, top_k=fetch_k)

        # BM25 retrieval
        bm25_results: list[dict] = []
        if self.bm25 and self.chunks:
            scores = self.bm25.get_scores(query.lower().split())
            top_indices = np.argsort(scores)[::-1][:fetch_k]
            for idx in top_indices:
                if scores[idx] > 0:
                    c = self.chunks[idx]
                    bm25_results.append(
                        {
                            "entry_id": c["chunk_id"],
                            "text": c["text"],
                            "source": c["source"],
                            "score": float(scores[idx]),
                        }
                    )

        merged = self._rrf(dense, bm25_results, top_k)
        return [
            RetrievalChunk(
                chunk_id=r["entry_id"],
                source=r["source"],
                text=r["text"],
                score=r["score"],
            )
            for r in merged
        ]

    @staticmethod
    def _rrf(
        dense: list[dict],
        bm25: list[dict],
        top_k: int,
        k: int = 60,
    ) -> list[dict]:
        """Reciprocal Rank Fusion."""
        scores: dict[str, float] = {}
        data: dict[str, dict] = {}
        for rank, r in enumerate(dense):
            doc_id = r["entry_id"]
            scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank + 1)
            data.setdefault(doc_id, r)
        for rank, r in enumerate(bm25):
            doc_id = r["entry_id"]
            scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank + 1)
            data.setdefault(doc_id, r)
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return [{**data[doc_id], "score": sc} for doc_id, sc in ranked]
