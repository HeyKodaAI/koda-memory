"""Tests for the memory system — models, storage, manager."""

from __future__ import annotations

import os
import tempfile
from datetime import datetime, timedelta, timezone

import pytest

from koda_memory.models import (
    CategoryControl,
    ContextWindow,
    MemoryCategory,
    MemoryItem,
    MemoryQuery,
    MemoryStats,
    MemoryTier,
)
from koda_memory.storage import MemoryStorage
from koda_memory.manager import MemoryManager


# ============================================================
# Model tests
# ============================================================

class TestMemoryModels:
    def test_memory_item_defaults(self):
        item = MemoryItem(content="test fact")
        assert item.tier == MemoryTier.LONG_TERM
        assert item.category == MemoryCategory.FACT
        assert item.confidence == 1.0
        assert item.access_count == 0
        assert item.id  # auto-generated

    def test_memory_item_not_expired(self):
        item = MemoryItem(content="test")
        assert not item.is_expired()

    def test_memory_item_expired(self):
        past = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
        item = MemoryItem(content="test", expires_at=past)
        assert item.is_expired()

    def test_memory_item_not_yet_expired(self):
        future = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()
        item = MemoryItem(content="test", expires_at=future)
        assert not item.is_expired()

    def test_memory_item_touch(self):
        item = MemoryItem(content="test")
        old_updated = item.updated_at
        item.touch()
        assert item.access_count == 1
        assert item.updated_at >= old_updated

    def test_memory_tiers(self):
        assert len(MemoryTier) == 4

    def test_memory_categories(self):
        assert len(MemoryCategory) == 11
        assert MemoryCategory.REFLECTION in MemoryCategory

    def test_context_window_empty(self):
        ctx = ContextWindow()
        assert ctx.to_prompt_section() == ""

    def test_context_window_formatting(self):
        items = [
            MemoryItem(key="timezone", content="UTC-5", category=MemoryCategory.PREFERENCE),
            MemoryItem(content="Works on Koda AI project", category=MemoryCategory.PROJECT),
        ]
        ctx = ContextWindow(items=items)
        prompt = ctx.to_prompt_section()
        assert "timezone" in prompt
        assert "UTC-5" in prompt
        assert "Koda AI" in prompt
        assert "[preference]" in prompt
        assert "[project]" in prompt

    def test_memory_query_defaults(self):
        q = MemoryQuery()
        assert q.limit == 50
        assert q.min_confidence == 0.0
        assert not q.include_expired


# ============================================================
# Storage tests
# ============================================================

class TestMemoryStorage:
    @pytest.fixture
    def storage(self, tmp_path):
        db = str(tmp_path / "test_memory.db")
        return MemoryStorage(db)

    def test_save_and_get(self, storage):
        item = MemoryItem(
            content="Mike's timezone is UTC-5",
            key="timezone",
            category=MemoryCategory.PREFERENCE,
        )
        storage.save(item)
        retrieved = storage.get(item.id)
        assert retrieved is not None
        assert retrieved.content == "Mike's timezone is UTC-5"
        assert retrieved.key == "timezone"

    def test_delete(self, storage):
        item = MemoryItem(content="test")
        storage.save(item)
        assert storage.delete(item.id)
        assert storage.get(item.id) is None

    def test_delete_nonexistent(self, storage):
        assert not storage.delete("nonexistent")

    def test_count(self, storage):
        assert storage.count() == 0
        storage.save(MemoryItem(content="a"))
        storage.save(MemoryItem(content="b"))
        assert storage.count() == 2

    def test_search_by_text(self, storage):
        storage.save(MemoryItem(content="Mike likes Python"))
        storage.save(MemoryItem(content="Mike likes TypeScript"))
        storage.save(MemoryItem(content="The sky is blue"))

        results = storage.search(MemoryQuery(text="Mike"))
        assert len(results) == 2

    def test_search_by_tier(self, storage):
        storage.save(MemoryItem(content="a", tier=MemoryTier.LONG_TERM))
        storage.save(MemoryItem(content="b", tier=MemoryTier.SHORT_TERM))
        storage.save(MemoryItem(content="c", tier=MemoryTier.LONG_TERM))

        results = storage.search(MemoryQuery(tier=MemoryTier.LONG_TERM))
        assert len(results) == 2

    def test_search_by_category(self, storage):
        storage.save(MemoryItem(content="a", category=MemoryCategory.PREFERENCE))
        storage.save(MemoryItem(content="b", category=MemoryCategory.PROJECT))

        results = storage.search(MemoryQuery(category=MemoryCategory.PREFERENCE))
        assert len(results) == 1

    def test_search_by_tags(self, storage):
        storage.save(MemoryItem(content="a", tags=["work", "python"]))
        storage.save(MemoryItem(content="b", tags=["personal"]))

        results = storage.search(MemoryQuery(tags=["python"]))
        assert len(results) == 1

    def test_search_excludes_expired(self, storage):
        past = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
        storage.save(MemoryItem(content="old", expires_at=past))
        storage.save(MemoryItem(content="current"))

        results = storage.search(MemoryQuery())
        assert len(results) == 1
        assert results[0].content == "current"

    def test_update_content(self, storage):
        item = MemoryItem(content="old value", key="test")
        storage.save(item)
        assert storage.update_content(item.id, "new value")
        updated = storage.get(item.id)
        assert updated.content == "new value"

    def test_delete_by_category(self, storage):
        storage.save(MemoryItem(content="a", category=MemoryCategory.HEALTH))
        storage.save(MemoryItem(content="b", category=MemoryCategory.HEALTH))
        storage.save(MemoryItem(content="c", category=MemoryCategory.FACT))

        deleted = storage.delete_by_category(MemoryCategory.HEALTH)
        assert deleted == 2
        assert storage.count() == 1

    def test_wipe_all(self, storage):
        storage.save(MemoryItem(content="a"))
        storage.save(MemoryItem(content="b"))
        deleted = storage.wipe_all()
        assert deleted == 2
        assert storage.count() == 0

    def test_get_stats(self, storage):
        storage.save(MemoryItem(content="a", tier=MemoryTier.LONG_TERM, category=MemoryCategory.FACT))
        storage.save(MemoryItem(content="b", tier=MemoryTier.SHORT_TERM, category=MemoryCategory.TASK))
        stats = storage.get_stats()
        assert stats.total_items == 2
        assert stats.by_tier["long_term"] == 1
        assert stats.by_tier["short_term"] == 1
        assert stats.by_category["fact"] == 1

    def test_category_controls(self, storage):
        controls = storage.get_category_controls()
        # All categories should be present with defaults
        assert len(controls) == len(MemoryCategory)
        assert all(c.enabled for c in controls)

    def test_set_category_control(self, storage):
        control = CategoryControl(
            category=MemoryCategory.HEALTH,
            enabled=False,
        )
        storage.set_category_control(control)
        assert not storage.is_category_enabled(MemoryCategory.HEALTH)
        assert storage.is_category_enabled(MemoryCategory.FACT)  # Others unaffected

    def test_export_all(self, storage):
        storage.save(MemoryItem(content="a"))
        storage.save(MemoryItem(content="b"))
        exported = storage.export_all()
        assert len(exported) == 2
        assert all(isinstance(e, dict) for e in exported)

    def test_list_all(self, storage):
        storage.save(MemoryItem(content="a"))
        storage.save(MemoryItem(content="b"))
        items = storage.list_all()
        assert len(items) == 2


# ============================================================
# Manager tests
# ============================================================

class TestMemoryManager:
    @pytest.fixture
    def manager(self, tmp_path):
        db = str(tmp_path / "test_memory.db")
        storage = MemoryStorage(db)
        return MemoryManager(storage)

    def test_remember_basic(self, manager):
        item = manager.remember("Mike's timezone is UTC-5", key="timezone")
        assert item is not None
        assert item.key == "timezone"

    def test_remember_auto_categorize(self, manager):
        item = manager.remember("Mike prefers dark mode and likes Python")
        assert item is not None
        assert item.category == MemoryCategory.PREFERENCE

    def test_remember_respects_privacy(self, manager):
        manager.set_control(CategoryControl(
            category=MemoryCategory.HEALTH,
            enabled=False,
        ))
        item = manager.remember(
            "Mike has allergies",
            category=MemoryCategory.HEALTH,
        )
        assert item is None

    def test_remember_with_expiry(self, manager):
        item = manager.remember(
            "Current session context",
            tier=MemoryTier.SHORT_TERM,
        )
        assert item is not None
        assert item.expires_at is not None

    def test_remember_long_term_no_expiry(self, manager):
        item = manager.remember(
            "Key fact",
            tier=MemoryTier.LONG_TERM,
        )
        assert item is not None
        assert item.expires_at is None

    def test_forget(self, manager):
        item = manager.remember("to be deleted")
        assert manager.forget(item.id)
        assert manager.get(item.id) is None

    def test_forget_category(self, manager):
        manager.remember("health a", category=MemoryCategory.HEALTH)
        manager.remember("health b", category=MemoryCategory.HEALTH)
        manager.remember("fact c", category=MemoryCategory.FACT)

        deleted = manager.forget_category(MemoryCategory.HEALTH)
        assert deleted == 2
        assert len(manager.recall()) == 1

    def test_wipe(self, manager):
        manager.remember("a")
        manager.remember("b")
        deleted = manager.wipe()
        assert deleted == 2

    def test_recall_text_search(self, manager):
        manager.remember("Python is great", key="lang")
        manager.remember("TypeScript is great", key="lang2")
        manager.remember("The weather is nice")

        results = manager.recall("Python")
        assert len(results) == 1
        assert "Python" in results[0].content

    def test_recall_by_tier(self, manager):
        manager.remember("a", tier=MemoryTier.LONG_TERM)
        manager.remember("b", tier=MemoryTier.EPISODIC)

        results = manager.recall(tier=MemoryTier.EPISODIC)
        assert len(results) == 1

    def test_update(self, manager):
        item = manager.remember("old", key="test")
        assert manager.update(item.id, "new")
        updated = manager.get(item.id)
        assert updated.content == "new"

    def test_build_context_empty(self, manager):
        ctx = manager.build_context()
        assert len(ctx.items) == 0
        assert ctx.to_prompt_section() == ""

    def test_build_context_with_items(self, manager):
        manager.remember("Timezone: UTC-5", key="timezone", tier=MemoryTier.LONG_TERM)
        manager.remember("Uses Python daily", key="language", tier=MemoryTier.LONG_TERM)
        manager.remember("Had meeting with Alice", tier=MemoryTier.EPISODIC)

        ctx = manager.build_context()
        assert len(ctx.items) >= 2
        prompt = ctx.to_prompt_section()
        assert "timezone" in prompt.lower() or "UTC-5" in prompt

    def test_build_context_respects_limits(self, manager):
        for i in range(30):
            manager.remember(f"Fact number {i}", key=f"fact_{i}")

        ctx = manager.build_context(max_items=5)
        assert len(ctx.items) <= 5

    def test_set_control_disables_and_deletes(self, manager):
        manager.remember("health info", category=MemoryCategory.HEALTH)
        assert len(manager.recall(category=MemoryCategory.HEALTH)) == 1

        manager.set_control(CategoryControl(
            category=MemoryCategory.HEALTH,
            enabled=False,
        ))
        assert len(manager.recall(category=MemoryCategory.HEALTH)) == 0

    def test_cleanup_expired(self, manager):
        # Store an already-expired item directly via storage
        past = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
        item = MemoryItem(content="expired", expires_at=past)
        manager.storage.save(item)

        # Store a valid item
        manager.remember("valid")

        deleted = manager.cleanup_expired()
        assert deleted == 1
        assert manager.storage.count() == 1

    def test_stats(self, manager):
        manager.remember("a", tier=MemoryTier.LONG_TERM)
        manager.remember("b", tier=MemoryTier.EPISODIC)
        stats = manager.stats()
        assert stats.total_items == 2

    def test_export(self, manager):
        manager.remember("a")
        manager.remember("b")
        exported = manager.export_all()
        assert len(exported) == 2

    def test_auto_categorize_financial(self, manager):
        item = manager.remember("Monthly budget is $5000 for expenses")
        assert item.category == MemoryCategory.FINANCIAL

    def test_auto_categorize_project(self, manager):
        item = manager.remember("Project deadline is March 15 for the release milestone")
        assert item.category == MemoryCategory.PROJECT

    def test_get_controls(self, manager):
        controls = manager.get_controls()
        assert len(controls) == len(MemoryCategory)
