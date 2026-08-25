"""Semantic memory engine — vector embeddings for Koda's memory system.

Generates embeddings for memory items and provides cosine-similarity
search so Koda can recall memories by *meaning*, not just keywords.

Architecture:
    - Embeddings generated via OpenAI text-embedding-3-small (1536 dims)
    - Stored as packed float32 BLOBs in SQLite alongside memory items
    - In-memory numpy index rebuilt on startup for fast cosine search
    - Graceful degradation: if embeddings unavailable, falls back to keyword search

Usage:
    engine = EmbeddingEngine(db_path="memory.db")
    await engine.init(api_key="sk-...")

    # Generate + store embedding for a new memory
    await engine.embed_memory(item_id, "User prefers dark mode")

    # Semantic search: find memories similar to a query
    results = await engine.search("what theme does the user like?", top_k=10)
    # → [(item_id, similarity_score), ...]
"""

from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
import struct
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger("koda_memory.embeddings")

# Embedding dimensions for supported models
_MODEL_DIMS = {
    # OpenAI models
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
    # Local (Ollama) models — zero API cost
    "nomic-embed-text": 768,
    "mxbai-embed-large": 1024,
    "all-minilm": 384,
    "snowflake-arctic-embed": 1024,
}

_DEFAULT_MODEL = "text-embedding-3-small"
_DEFAULT_DIMS = 1536

# Batch size for embedding API calls
_EMBED_BATCH_SIZE = 100

# Minimum similarity threshold for search results
_MIN_SIMILARITY = 0.15


@dataclass
class SemanticResult:
    """A single semantic search result."""
    item_id: str
    similarity: float  # 0.0 – 1.0 cosine similarity


class EmbeddingEngine:
    """Vector embedding engine for semantic memory search.

    Manages embedding generation (via OpenAI API), storage (SQLite BLOB),
    and in-memory cosine search (via numpy).  Designed to be resilient:
    if the API is unreachable or numpy isn't installed, everything
    degrades gracefully to keyword-only search.
    """

    def __init__(self, db_path: str = "memory.db") -> None:
        self._db_path = db_path
        self._conn: Optional[sqlite3.Connection] = None
        self._model = _DEFAULT_MODEL
        self._dims = _DEFAULT_DIMS
        self._api_key: str = ""
        self._base_url: str = ""  # For custom endpoints
        self._client: Any = None  # openai.AsyncOpenAI
        self._ready = False

        # In-memory index: rebuilt from DB on init
        self._index_ids: list[str] = []          # item_id at each row
        self._index_matrix: Any = None            # numpy float32 matrix (N x dims)
        self._np: Any = None                      # numpy module ref

    @property
    def ready(self) -> bool:
        """True if engine can generate embeddings (API client available)."""
        return self._ready

    @property
    def index_size(self) -> int:
        """Number of embeddings currently in the in-memory index."""
        return len(self._index_ids)

    @property
    def model(self) -> str:
        return self._model

    @property
    def dims(self) -> int:
        return self._dims

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    async def init(
        self,
        api_key: str = "",
        base_url: str = "",
        model: str = "",
    ) -> bool:
        """Initialize the embedding engine.

        1. Import numpy (optional — degrades without it)
        2. Create SQLite table for embeddings
        3. Build OpenAI client for embedding generation
        4. Load existing embeddings into in-memory index

        Returns True if fully operational, False if degraded.
        """
        # 1. Try to import numpy
        try:
            import numpy as np
            self._np = np
        except ImportError:
            logger.warning("numpy not installed — semantic search disabled")
            return False

        # 2. Set up embedding model
        if model and model in _MODEL_DIMS:
            self._model = model
            self._dims = _MODEL_DIMS[model]

        # 3. Connect to SQLite and create table
        self._conn = sqlite3.connect(self._db_path)
        self._conn.row_factory = sqlite3.Row
        self._create_tables()

        # 4. Build embedding client (OpenAI API or Ollama-compatible)
        self._api_key = api_key
        self._base_url = base_url

        # Detect local/Ollama models — they don't need a real API key
        _is_local = (
            self._model in ("nomic-embed-text", "mxbai-embed-large", "all-minilm", "snowflake-arctic-embed")
            or (base_url and "11434" in base_url)
            or api_key == "ollama"
        )

        if api_key or _is_local:
            try:
                from openai import AsyncOpenAI
                client_kwargs: dict[str, Any] = {
                    "api_key": api_key if api_key and api_key != "ollama" else "ollama",
                }
                if base_url:
                    client_kwargs["base_url"] = base_url
                elif _is_local:
                    client_kwargs["base_url"] = "http://localhost:11434/v1"
                self._client = AsyncOpenAI(**client_kwargs)
                self._ready = True
                _mode = "LOCAL (Ollama)" if _is_local else "CLOUD (OpenAI)"
                logger.info(
                    "Embedding engine ready [%s]: model=%s dims=%d",
                    _mode, self._model, self._dims,
                )
            except Exception:
                logger.warning("Failed to create embedding client", exc_info=True)
                self._ready = False
        else:
            logger.info("No embedding API key — semantic search in read-only mode")
            self._ready = False

        # 5. Load existing embeddings into memory
        loaded = self._load_index()
        logger.info("Loaded %d embeddings into in-memory index", loaded)

        return self._ready

    def _create_tables(self) -> None:
        """Create the embeddings storage table."""
        if not self._conn:
            return
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS memory_embeddings (
                item_id TEXT PRIMARY KEY,
                model TEXT NOT NULL,
                dims INTEGER NOT NULL,
                embedding BLOB NOT NULL,
                created_at REAL NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_embedding_model
                ON memory_embeddings(model);
        """)
        self._conn.commit()

    def _load_index(self) -> int:
        """Load all embeddings from SQLite into the in-memory numpy index."""
        if not self._conn or not self._np:
            return 0

        np = self._np
        rows = self._conn.execute(
            "SELECT item_id, embedding, dims FROM memory_embeddings WHERE model = ?",
            (self._model,),
        ).fetchall()

        if not rows:
            self._index_ids = []
            self._index_matrix = np.empty((0, self._dims), dtype=np.float32)
            return 0

        ids: list[str] = []
        vectors: list[Any] = []

        for row in rows:
            item_id = row["item_id"]
            blob = row["embedding"]
            dims = row["dims"]

            # Unpack float32 BLOB
            expected_bytes = dims * 4
            if len(blob) != expected_bytes:
                logger.warning("Skipping embedding %s: size mismatch", item_id)
                continue

            vec = np.frombuffer(blob, dtype=np.float32).copy()
            ids.append(item_id)
            vectors.append(vec)

        if vectors:
            self._index_ids = ids
            self._index_matrix = np.vstack(vectors)
        else:
            self._index_ids = []
            self._index_matrix = np.empty((0, self._dims), dtype=np.float32)

        return len(self._index_ids)

    # ------------------------------------------------------------------
    # Embedding generation
    # ------------------------------------------------------------------

    async def generate_embedding(self, text: str) -> Optional[list[float]]:
        """Generate an embedding vector for a text string.

        Returns None if the API call fails or client isn't available.
        """
        if not self._client:
            return None

        # Truncate to ~8000 tokens worth (~32000 chars) — model limit is 8191 tokens
        truncated = text[:32000]

        try:
            response = await self._client.embeddings.create(
                model=self._model,
                input=truncated,
            )
            return response.data[0].embedding
        except Exception:
            logger.warning("Embedding generation failed", exc_info=True)
            return None

    async def generate_embeddings_batch(
        self, texts: list[str],
    ) -> list[Optional[list[float]]]:
        """Generate embeddings for multiple texts in batches.

        Returns a list of embeddings (or None for failures), same length as input.
        """
        if not self._client:
            return [None] * len(texts)

        results: list[Optional[list[float]]] = [None] * len(texts)

        for batch_start in range(0, len(texts), _EMBED_BATCH_SIZE):
            batch_end = min(batch_start + _EMBED_BATCH_SIZE, len(texts))
            batch_texts = [t[:32000] for t in texts[batch_start:batch_end]]

            try:
                response = await self._client.embeddings.create(
                    model=self._model,
                    input=batch_texts,
                )
                for i, item in enumerate(response.data):
                    results[batch_start + i] = item.embedding
            except Exception:
                logger.warning(
                    "Batch embedding failed for items %d-%d",
                    batch_start, batch_end, exc_info=True,
                )

        return results

    # ------------------------------------------------------------------
    # Storage
    # ------------------------------------------------------------------

    def store_embedding(self, item_id: str, embedding: list[float]) -> None:
        """Store an embedding in SQLite and update the in-memory index."""
        if not self._conn or not self._np:
            return

        np = self._np

        # Pack as float32 BLOB
        blob = struct.pack(f"{len(embedding)}f", *embedding)

        self._conn.execute(
            """INSERT OR REPLACE INTO memory_embeddings
               (item_id, model, dims, embedding, created_at)
               VALUES (?, ?, ?, ?, ?)""",
            (item_id, self._model, len(embedding), blob, time.time()),
        )
        self._conn.commit()

        # Update in-memory index
        vec = np.array(embedding, dtype=np.float32).reshape(1, -1)

        if item_id in self._index_ids:
            # Replace existing
            idx = self._index_ids.index(item_id)
            self._index_matrix[idx] = vec[0]
        else:
            # Append new
            self._index_ids.append(item_id)
            if self._index_matrix is not None and self._index_matrix.shape[0] > 0:
                self._index_matrix = np.vstack([self._index_matrix, vec])
            else:
                self._index_matrix = vec

    def remove_embedding(self, item_id: str) -> None:
        """Remove an embedding from storage and the in-memory index."""
        if not self._conn:
            return

        self._conn.execute(
            "DELETE FROM memory_embeddings WHERE item_id = ?", (item_id,)
        )
        self._conn.commit()

        # Update in-memory index
        if item_id in self._index_ids:
            idx = self._index_ids.index(item_id)
            self._index_ids.pop(idx)
            if self._np is not None and self._index_matrix is not None:
                self._index_matrix = self._np.delete(self._index_matrix, idx, axis=0)

    def has_embedding(self, item_id: str) -> bool:
        """Check if an embedding exists for a memory item."""
        return item_id in self._index_ids

    # ------------------------------------------------------------------
    # Semantic search
    # ------------------------------------------------------------------

    async def search(
        self,
        query: str,
        top_k: int = 20,
        min_similarity: float = _MIN_SIMILARITY,
    ) -> list[SemanticResult]:
        """Find the most semantically similar memories to a query.

        1. Generate embedding for the query text
        2. Compute cosine similarity against all stored embeddings
        3. Return top-k results above the minimum threshold

        Returns empty list if embeddings are unavailable.
        """
        if not self._np or self._index_matrix is None or self._index_matrix.shape[0] == 0:
            return []

        # Generate query embedding
        query_vec = await self.generate_embedding(query)
        if query_vec is None:
            return []

        return self.search_by_vector(query_vec, top_k=top_k, min_similarity=min_similarity)

    def search_by_vector(
        self,
        query_vec: list[float],
        top_k: int = 20,
        min_similarity: float = _MIN_SIMILARITY,
    ) -> list[SemanticResult]:
        """Search using a pre-computed query vector (no API call needed).

        Useful when you've already generated the embedding elsewhere.
        """
        if not self._np or self._index_matrix is None or self._index_matrix.shape[0] == 0:
            return []

        np = self._np

        # Convert to numpy
        q = np.array(query_vec, dtype=np.float32)

        # Cosine similarity = dot(a, b) / (||a|| * ||b||)
        # Normalize query
        q_norm = np.linalg.norm(q)
        if q_norm < 1e-10:
            return []
        q = q / q_norm

        # Normalize matrix rows
        norms = np.linalg.norm(self._index_matrix, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)  # Avoid division by zero
        normed_matrix = self._index_matrix / norms

        # Compute similarities (matrix × vector)
        similarities = normed_matrix @ q

        # Get top-k indices above threshold
        mask = similarities >= min_similarity
        valid_indices = np.where(mask)[0]

        if len(valid_indices) == 0:
            return []

        # Sort by similarity descending
        sorted_valid = valid_indices[np.argsort(similarities[valid_indices])[::-1]]
        top_indices = sorted_valid[:top_k]

        results: list[SemanticResult] = []
        for idx in top_indices:
            results.append(SemanticResult(
                item_id=self._index_ids[idx],
                similarity=float(similarities[idx]),
            ))

        return results

    # ------------------------------------------------------------------
    # Embed + store (convenience)
    # ------------------------------------------------------------------

    async def embed_memory(self, item_id: str, text: str) -> bool:
        """Generate and store an embedding for a memory item.

        Returns True if successful, False on failure.
        """
        if not self._ready:
            return False

        embedding = await self.generate_embedding(text)
        if embedding is None:
            return False

        self.store_embedding(item_id, embedding)
        return True

    async def embed_memory_batch(
        self, items: list[tuple[str, str]],
    ) -> int:
        """Generate and store embeddings for multiple memory items.

        Args:
            items: list of (item_id, text) tuples

        Returns count of successfully embedded items.
        """
        if not self._ready or not items:
            return 0

        # Filter out items that already have embeddings
        to_embed = [(iid, text) for iid, text in items if not self.has_embedding(iid)]
        if not to_embed:
            return 0

        ids = [iid for iid, _ in to_embed]
        texts = [text for _, text in to_embed]

        embeddings = await self.generate_embeddings_batch(texts)

        count = 0
        for item_id, embedding in zip(ids, embeddings):
            if embedding is not None:
                self.store_embedding(item_id, embedding)
                count += 1

        return count

    # ------------------------------------------------------------------
    # Backfill
    # ------------------------------------------------------------------

    async def backfill_from_storage(self, storage: Any) -> dict[str, int]:
        """Backfill embeddings for all existing memories that don't have one.

        Args:
            storage: MemoryStorage instance to read memories from

        Returns: {"total": N, "embedded": M, "skipped": S, "failed": F}
        """
        if not self._ready:
            return {"total": 0, "embedded": 0, "skipped": 0, "failed": 0}

        from koda_memory.models import MemoryQuery

        all_items = storage.search(MemoryQuery(limit=10000, include_expired=True))

        to_embed: list[tuple[str, str]] = []
        skipped = 0

        for item in all_items:
            if self.has_embedding(item.id):
                skipped += 1
                continue
            # Combine key + content for richer embedding
            text = f"{item.key}: {item.content}" if item.key else item.content
            to_embed.append((item.id, text))

        embedded = await self.embed_memory_batch(to_embed)
        failed = len(to_embed) - embedded

        result = {
            "total": len(all_items),
            "embedded": embedded,
            "skipped": skipped,
            "failed": failed,
        }
        logger.info("Backfill complete: %s", result)
        return result

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------

    def cleanup_orphaned(self, valid_ids: set[str]) -> int:
        """Remove embeddings for deleted memory items.

        Args:
            valid_ids: Set of item IDs that still exist in memory storage

        Returns count of orphaned embeddings removed.
        """
        orphaned = [iid for iid in self._index_ids if iid not in valid_ids]
        for iid in orphaned:
            self.remove_embedding(iid)
        if orphaned:
            logger.info("Cleaned up %d orphaned embeddings", len(orphaned))
        return len(orphaned)

    def get_stats(self) -> dict[str, Any]:
        """Get embedding engine statistics."""
        return {
            "ready": self._ready,
            "model": self._model,
            "dims": self._dims,
            "index_size": len(self._index_ids),
            "has_numpy": self._np is not None,
            "has_client": self._client is not None,
        }
