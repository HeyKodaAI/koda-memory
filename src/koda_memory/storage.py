"""SQLite storage for the memory system."""

from __future__ import annotations

import json
import logging
import sqlite3
from typing import Optional

from koda_memory.models import (
    CategoryControl,
    MemoryCategory,
    MemoryItem,
    MemoryQuery,
    MemoryStats,
    MemoryTier,
)

logger = logging.getLogger("koda_memory.storage")


class MemoryStorage:
    """SQLite-backed storage for memory items and privacy controls."""

    def __init__(self, db_path: str = "memory.db") -> None:
        self._db_path = db_path
        self._conn = sqlite3.connect(db_path)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._create_tables()

    def _create_tables(self) -> None:
        # Step 1: Create tables (without new-column indexes)
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS memory_items (
                id TEXT PRIMARY KEY,
                tier TEXT NOT NULL,
                category TEXT NOT NULL,
                key TEXT NOT NULL DEFAULT '',
                content TEXT NOT NULL,
                source TEXT NOT NULL DEFAULT '',
                confidence REAL NOT NULL DEFAULT 1.0,
                tags TEXT NOT NULL DEFAULT '[]',
                metadata TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                expires_at TEXT,
                access_count INTEGER NOT NULL DEFAULT 0,
                pinned INTEGER NOT NULL DEFAULT 0
            );

            CREATE INDEX IF NOT EXISTS idx_memory_tier ON memory_items(tier);
            CREATE INDEX IF NOT EXISTS idx_memory_category ON memory_items(category);
            CREATE INDEX IF NOT EXISTS idx_memory_key ON memory_items(key);
            CREATE INDEX IF NOT EXISTS idx_memory_updated ON memory_items(updated_at);
            CREATE INDEX IF NOT EXISTS idx_memory_expires ON memory_items(expires_at);

            CREATE TABLE IF NOT EXISTS category_controls (
                category TEXT PRIMARY KEY,
                enabled INTEGER NOT NULL DEFAULT 1,
                auto_expire_days INTEGER
            );

            CREATE TABLE IF NOT EXISTS relationship_summary (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                summary TEXT NOT NULL DEFAULT '',
                updated_at TEXT NOT NULL,
                source_hash TEXT NOT NULL DEFAULT ''
            );
        """)
        self._conn.commit()

        # Step 2: Migrate existing databases — add pinned column if missing
        self._migrate_add_column("memory_items", "pinned", "INTEGER NOT NULL DEFAULT 0")

        # Step 3: Now safe to create index on pinned (column guaranteed to exist)
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_memory_pinned ON memory_items(pinned)"
        )
        self._conn.commit()

    def _migrate_add_column(self, table: str, column: str, typedef: str) -> None:
        """Add a column if it doesn't exist (safe migration for existing DBs)."""
        try:
            cols = [r[1] for r in self._conn.execute(f"PRAGMA table_info({table})").fetchall()]
            if column not in cols:
                self._conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {typedef}")
                self._conn.commit()
                logger.info("Migrated %s: added column %s", table, column)
        except Exception:
            logger.debug("Migration check for %s.%s skipped", table, column, exc_info=True)

    # ------------------------------------------------------------------
    # Memory CRUD
    # ------------------------------------------------------------------

    def save(self, item: MemoryItem) -> str:
        """Insert or replace a memory item."""
        self._conn.execute(
            """INSERT OR REPLACE INTO memory_items
               (id, tier, category, key, content, source, confidence,
                tags, metadata, created_at, updated_at, expires_at,
                access_count, pinned)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                item.id, item.tier.value, item.category.value,
                item.key, item.content, item.source, item.confidence,
                json.dumps(item.tags), json.dumps(item.metadata),
                item.created_at, item.updated_at, item.expires_at,
                item.access_count, 1 if item.pinned else 0,
            ),
        )
        self._conn.commit()
        return item.id

    def get(self, item_id: str) -> Optional[MemoryItem]:
        """Retrieve a single memory item by ID."""
        row = self._conn.execute(
            "SELECT * FROM memory_items WHERE id = ?", (item_id,)
        ).fetchone()
        return self._row_to_item(row) if row else None

    def delete(self, item_id: str) -> bool:
        """Delete a memory item."""
        cur = self._conn.execute(
            "DELETE FROM memory_items WHERE id = ?", (item_id,)
        )
        self._conn.commit()
        return cur.rowcount > 0

    def delete_by_category(self, category: MemoryCategory) -> int:
        """Delete all memories in a category. Returns count deleted."""
        cur = self._conn.execute(
            "DELETE FROM memory_items WHERE category = ?", (category.value,)
        )
        self._conn.commit()
        return cur.rowcount

    def wipe_all(self) -> int:
        """Delete ALL memory items. Returns count deleted."""
        cur = self._conn.execute("DELETE FROM memory_items")
        self._conn.commit()
        return cur.rowcount

    def update_content(self, item_id: str, content: str) -> bool:
        """Update just the content of a memory item."""
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc).isoformat()
        cur = self._conn.execute(
            "UPDATE memory_items SET content = ?, updated_at = ? WHERE id = ?",
            (content, now, item_id),
        )
        self._conn.commit()
        return cur.rowcount > 0

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def search(self, query: MemoryQuery) -> list[MemoryItem]:
        """Search memory items with filters."""
        conditions: list[str] = []
        params: list = []

        if not query.include_expired:
            from datetime import datetime, timezone
            now_iso = datetime.now(timezone.utc).isoformat()
            conditions.append("(expires_at IS NULL OR expires_at > ?)")
            params.append(now_iso)

        if query.tier:
            conditions.append("tier = ?")
            params.append(query.tier.value)

        if query.category:
            conditions.append("category = ?")
            params.append(query.category.value)

        if query.min_confidence > 0:
            conditions.append("confidence >= ?")
            params.append(query.min_confidence)

        if query.text:
            conditions.append("(content LIKE ? OR key LIKE ?)")
            like = f"%{query.text}%"
            params.extend([like, like])

        if query.tags:
            # Match any tag
            tag_conds = []
            for tag in query.tags:
                tag_conds.append("tags LIKE ?")
                params.append(f'%"{tag}"%')
            conditions.append(f"({' OR '.join(tag_conds)})")

        where = " AND ".join(conditions) if conditions else "1=1"
        sql = f"""
            SELECT * FROM memory_items
            WHERE {where}
            ORDER BY updated_at DESC
            LIMIT ?
        """
        params.append(query.limit)

        rows = self._conn.execute(sql, params).fetchall()
        return [self._row_to_item(r) for r in rows]

    def list_all(self, limit: int = 200) -> list[MemoryItem]:
        """List all non-expired memories."""
        return self.search(MemoryQuery(limit=limit))

    def count(self) -> int:
        """Total number of memory items."""
        row = self._conn.execute("SELECT COUNT(*) FROM memory_items").fetchone()
        return row[0] if row else 0

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def get_stats(self) -> MemoryStats:
        """Get summary statistics."""
        total = self.count()

        by_tier: dict[str, int] = {}
        for row in self._conn.execute(
            "SELECT tier, COUNT(*) as cnt FROM memory_items GROUP BY tier"
        ).fetchall():
            by_tier[row["tier"]] = row["cnt"]

        by_category: dict[str, int] = {}
        for row in self._conn.execute(
            "SELECT category, COUNT(*) as cnt FROM memory_items GROUP BY category"
        ).fetchall():
            by_category[row["category"]] = row["cnt"]

        oldest = self._conn.execute(
            "SELECT created_at FROM memory_items ORDER BY created_at ASC LIMIT 1"
        ).fetchone()
        newest = self._conn.execute(
            "SELECT created_at FROM memory_items ORDER BY created_at DESC LIMIT 1"
        ).fetchone()

        return MemoryStats(
            total_items=total,
            by_tier=by_tier,
            by_category=by_category,
            oldest_item=oldest["created_at"] if oldest else None,
            newest_item=newest["created_at"] if newest else None,
        )

    # ------------------------------------------------------------------
    # Category controls (privacy)
    # ------------------------------------------------------------------

    def get_category_controls(self) -> list[CategoryControl]:
        """Get all category privacy controls."""
        rows = self._conn.execute("SELECT * FROM category_controls").fetchall()
        controls = {r["category"]: r for r in rows}

        # Return all categories, with defaults for missing ones
        result = []
        for cat in MemoryCategory:
            if cat.value in controls:
                row = controls[cat.value]
                result.append(CategoryControl(
                    category=cat,
                    enabled=bool(row["enabled"]),
                    auto_expire_days=row["auto_expire_days"],
                ))
            else:
                result.append(CategoryControl(category=cat, enabled=True))
        return result

    def set_category_control(self, control: CategoryControl) -> None:
        """Set privacy control for a category."""
        self._conn.execute(
            """INSERT OR REPLACE INTO category_controls
               (category, enabled, auto_expire_days)
               VALUES (?, ?, ?)""",
            (
                control.category.value,
                1 if control.enabled else 0,
                control.auto_expire_days,
            ),
        )
        self._conn.commit()

    def is_category_enabled(self, category: MemoryCategory) -> bool:
        """Check if a category is enabled for memorization."""
        row = self._conn.execute(
            "SELECT enabled FROM category_controls WHERE category = ?",
            (category.value,),
        ).fetchone()
        return bool(row["enabled"]) if row else True  # Default: enabled

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export_all(self) -> list[dict]:
        """Export all memories as dicts (for JSON export)."""
        rows = self._conn.execute("SELECT * FROM memory_items ORDER BY created_at").fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Pinned memories
    # ------------------------------------------------------------------

    def set_pinned(self, item_id: str, pinned: bool) -> bool:
        """Pin or unpin a memory item."""
        cur = self._conn.execute(
            "UPDATE memory_items SET pinned = ? WHERE id = ?",
            (1 if pinned else 0, item_id),
        )
        self._conn.commit()
        return cur.rowcount > 0

    def get_pinned(self) -> list[MemoryItem]:
        """Get all pinned memory items (always included in context)."""
        rows = self._conn.execute(
            "SELECT * FROM memory_items WHERE pinned = 1 ORDER BY updated_at DESC"
        ).fetchall()
        return [self._row_to_item(r) for r in rows]

    # ------------------------------------------------------------------
    # Relationship summary
    # ------------------------------------------------------------------

    def get_relationship_summary(self) -> Optional[str]:
        """Get the cached relationship summary, or None if not yet built."""
        row = self._conn.execute(
            "SELECT summary FROM relationship_summary WHERE id = 1"
        ).fetchone()
        return row["summary"] if row else None

    def save_relationship_summary(self, summary: str, source_hash: str) -> None:
        """Save or update the relationship summary."""
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            """INSERT OR REPLACE INTO relationship_summary
               (id, summary, updated_at, source_hash)
               VALUES (1, ?, ?, ?)""",
            (summary, now, source_hash),
        )
        self._conn.commit()

    def get_relationship_summary_hash(self) -> str:
        """Get the source hash of the current relationship summary."""
        row = self._conn.execute(
            "SELECT source_hash FROM relationship_summary WHERE id = 1"
        ).fetchone()
        return row["source_hash"] if row else ""

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _row_to_item(row: sqlite3.Row) -> MemoryItem:
        # Handle both old (no pinned column) and new schema
        try:
            pinned = bool(row["pinned"])
        except (IndexError, KeyError):
            pinned = False

        return MemoryItem(
            id=row["id"],
            tier=MemoryTier(row["tier"]),
            category=MemoryCategory(row["category"]),
            key=row["key"],
            content=row["content"],
            source=row["source"],
            confidence=row["confidence"],
            tags=json.loads(row["tags"]),
            metadata=json.loads(row["metadata"]),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            expires_at=row["expires_at"],
            access_count=row["access_count"],
            pinned=pinned,
        )
