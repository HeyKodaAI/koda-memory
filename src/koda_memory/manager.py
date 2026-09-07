"""Memory manager — high-level API for the memory system.

Handles:
  - Storing new memories with category/privacy checks
  - Building context windows for LLM requests (hybrid keyword + semantic)
  - Pinned memories (always included in every context window)
  - Relationship summary (auto-condensed portrait of the user)
  - Expiry and cleanup
  - Memory extraction from conversations

Context injection pipeline (v3):
  1. Relationship summary — a stable, condensed paragraph about the user
     built from all long-term memories. Regenerated whenever memories change.
  2. Pinned memories — core facts the user or Koda has marked as essential.
     Always included regardless of message content.
  3. Retrieved memories — dynamic, message-dependent selection using
     hybrid keyword + semantic search with multi-factor scoring.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import math
import re
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Optional

from koda_memory.models import (
    CategoryControl,
    ContextWindow,
    EmbeddingStats,
    MemoryCategory,
    MemoryItem,
    MemoryQuery,
    MemoryStats,
    MemoryTier,
)
from koda_memory.storage import MemoryStorage

if TYPE_CHECKING:
    from koda_memory.embeddings import EmbeddingEngine

logger = logging.getLogger("koda_memory.manager")

# Default expiry per tier
_TIER_EXPIRY_DAYS = {
    MemoryTier.SHORT_TERM: 1,       # 24 hours
    MemoryTier.MEDIUM_TERM: 30,     # 30 days
    MemoryTier.LONG_TERM: None,     # Never expires
    MemoryTier.EPISODIC: None,      # Never expires
}

# Keywords → category mapping for auto-categorization
_CATEGORY_HINTS: dict[MemoryCategory, set[str]] = {
    MemoryCategory.PREFERENCE: {
        "prefer", "prefers", "preferred", "preference", "favorite",
        "like", "likes", "dislike", "dislikes", "style",
        "timezone", "language", "format", "mode", "dark", "light",
    },
    MemoryCategory.PROJECT: {
        "project", "deadline", "milestone", "sprint", "release",
        "roadmap", "goal", "objective",
    },
    MemoryCategory.CONTACT: {
        "colleague", "manager", "team", "client", "friend",
        "phone",
    },
    MemoryCategory.TASK: {
        "task", "todo", "remind", "recurring",
    },
    MemoryCategory.PATTERN: {
        "usually", "always", "pattern", "habit",
        "routine", "tends",
    },
    MemoryCategory.HEALTH: {
        "health", "medical", "doctor", "medication", "allergy",
        "allergies", "condition", "symptoms",
    },
    MemoryCategory.FINANCIAL: {
        "budget", "salary", "expense", "expenses", "investment",
        "bank", "tax", "income", "financial", "money", "cost",
    },
}

# Semantic search weight in hybrid scoring
_SEMANTIC_BOOST = 2.0  # Additive boost for high-similarity semantic matches

# Relationship summary config
_SUMMARY_MAX_MEMORIES = 200  # Max memories to feed into summary generation
_SUMMARY_MAX_LENGTH = 600    # Max characters for the relationship summary


class MemoryManager:
    """High-level memory operations with pinned memories and relationship summary."""

    def __init__(self, storage: MemoryStorage) -> None:
        self._storage = storage
        self._embeddings: Optional[EmbeddingEngine] = None
        self._on_remember_hooks: list = []  # [(callback, is_async), ...]
        # Cached relationship summary (rebuilt when memories change)
        self._relationship_summary: str = ""
        self._relationship_hash: str = ""
        self._load_cached_summary()
        self._ensure_summary_fresh()

    def _load_cached_summary(self) -> None:
        """Load cached relationship summary from storage on startup."""
        try:
            cached = self._storage.get_relationship_summary()
            if cached:
                self._relationship_summary = cached
                self._relationship_hash = self._storage.get_relationship_summary_hash()
                logger.debug("Loaded cached relationship summary (%d chars)",
                             len(self._relationship_summary))
        except Exception:
            logger.debug("No cached relationship summary found", exc_info=True)

    @property
    def storage(self) -> MemoryStorage:
        return self._storage

    @property
    def embeddings(self) -> Optional[EmbeddingEngine]:
        return self._embeddings

    @property
    def relationship_summary(self) -> str:
        return self._ensure_summary_fresh()

    def set_embedding_engine(self, engine: EmbeddingEngine) -> None:
        """Attach an embedding engine for semantic search."""
        self._embeddings = engine
        logger.info(
            "Semantic search enabled: model=%s, index=%d embeddings",
            engine.model, engine.index_size,
        )

    def add_on_remember_hook(self, callback, *, is_async: bool = True) -> None:
        """Register a callback to be called after a memory is stored."""
        self._on_remember_hooks.append((callback, is_async))

    def _fire_on_remember(self, item: MemoryItem) -> None:
        """Fire all registered on_remember hooks (best-effort)."""
        for callback, is_async in self._on_remember_hooks:
            try:
                if is_async:
                    try:
                        loop = asyncio.get_running_loop()
                        loop.create_task(callback(item))
                    except RuntimeError:
                        pass
                else:
                    callback(item)
            except Exception:
                logger.debug("on_remember hook failed", exc_info=True)

    def _schedule_embedding(self, item: MemoryItem) -> None:
        """Schedule async embedding generation for a new memory."""
        if self._embeddings and self._embeddings.ready:
            text = f"{item.key}: {item.content}" if item.key else item.content
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(self._embeddings.embed_memory(item.id, text))
            except RuntimeError:
                pass

    # ------------------------------------------------------------------
    # Store
    # ------------------------------------------------------------------

    def remember(
        self,
        content: str,
        *,
        key: str = "",
        tier: MemoryTier = MemoryTier.LONG_TERM,
        category: Optional[MemoryCategory] = None,
        source: str = "",
        confidence: float = 1.0,
        tags: Optional[list[str]] = None,
        metadata: Optional[dict] = None,
        pinned: bool = False,
    ) -> Optional[MemoryItem]:
        """Store a new memory, respecting privacy controls.

        Returns the stored item, or None if the category is disabled.
        Also schedules async embedding generation and marks summary as stale.
        """
        if category is None:
            category = self._guess_category(content)

        if not self._storage.is_category_enabled(category):
            logger.info("Category %s is disabled, not storing memory", category.value)
            return None

        now = datetime.now(timezone.utc)
        expires_at = None
        expiry_days = _TIER_EXPIRY_DAYS.get(tier)

        controls = {c.category: c for c in self._storage.get_category_controls()}
        if category in controls and controls[category].auto_expire_days is not None:
            expiry_days = controls[category].auto_expire_days

        if expiry_days is not None:
            expires_at = (now + timedelta(days=expiry_days)).isoformat()

        item = MemoryItem(
            tier=tier,
            category=category,
            key=key,
            content=content,
            source=source,
            confidence=confidence,
            tags=tags or [],
            metadata=metadata or {},
            expires_at=expires_at,
            pinned=pinned,
        )

        self._storage.save(item)
        logger.debug("Stored memory %s [%s/%s]: %s",
                      item.id, tier.value, category.value, key or content[:50])

        self._schedule_embedding(item)
        self._fire_on_remember(item)

        # Mark relationship summary as potentially stale for long-term items
        if tier in (MemoryTier.LONG_TERM, MemoryTier.EPISODIC):
            self._invalidate_summary()

        return item

    def update(self, item_id: str, content: str) -> bool:
        """Update an existing memory's content."""
        success = self._storage.update_content(item_id, content)
        if success:
            if self._embeddings and self._embeddings.ready:
                item = self._storage.get(item_id)
                if item:
                    text = f"{item.key}: {content}" if item.key else content
                    try:
                        loop = asyncio.get_running_loop()
                        loop.create_task(self._embeddings.embed_memory(item_id, text))
                    except RuntimeError:
                        pass
            self._invalidate_summary()
        return success

    def forget(self, item_id: str) -> bool:
        """Delete a specific memory and its embedding."""
        success = self._storage.delete(item_id)
        if success:
            if self._embeddings:
                self._embeddings.remove_embedding(item_id)
            self._invalidate_summary()
        return success

    def forget_category(self, category: MemoryCategory) -> int:
        """Delete all memories in a category and their embeddings."""
        items = self._storage.search(MemoryQuery(
            category=category, limit=10000, include_expired=True,
        ))
        deleted = self._storage.delete_by_category(category)
        if self._embeddings and deleted:
            for item in items:
                self._embeddings.remove_embedding(item.id)
        if deleted:
            self._invalidate_summary()
        return deleted

    def wipe(self) -> int:
        """Delete ALL memories and embeddings."""
        deleted = self._storage.wipe_all()
        if self._embeddings:
            self._embeddings.cleanup_orphaned(set())
        self._relationship_summary = ""
        self._relationship_hash = ""
        self._storage.save_relationship_summary("", "")
        return deleted

    # ------------------------------------------------------------------
    # Pinned memories
    # ------------------------------------------------------------------

    def pin_memory(self, item_id: str) -> bool:
        """Pin a memory so it's always included in context."""
        return self._storage.set_pinned(item_id, True)

    def unpin_memory(self, item_id: str) -> bool:
        """Unpin a memory."""
        return self._storage.set_pinned(item_id, False)

    def get_pinned(self) -> list[MemoryItem]:
        """Get all pinned memories."""
        return self._storage.get_pinned()

    # ------------------------------------------------------------------
    # Relationship summary
    # ------------------------------------------------------------------

    def _invalidate_summary(self) -> None:
        """Mark the relationship summary as potentially stale.

        The summary will be lazily rebuilt on the next build_context() call
        if the memory fingerprint has actually changed.
        """
        self._relationship_summary = ""
        self._relationship_hash = ""
        self._storage.save_relationship_summary("", "")

    def _compute_memory_fingerprint(self) -> str:
        """Compute a hash of all long-term + pinned memories.

        Used to detect when the summary needs rebuilding.
        """
        items = self._storage.search(MemoryQuery(
            tier=MemoryTier.LONG_TERM, limit=_SUMMARY_MAX_MEMORIES,
        ))
        items.extend(self._storage.search(MemoryQuery(tier=MemoryTier.EPISODIC, limit=50)))
        # Also include pinned items from any tier
        pinned = self._storage.get_pinned()
        seen = {i.id for i in items}
        for p in pinned:
            if p.id not in seen:
                items.append(p)
                seen.add(p.id)

        # Sort by ID for deterministic hash
        items.sort(key=lambda i: i.id)
        content = "|".join(f"{i.id}:{i.content[:100]}:{i.updated_at}" for i in items)
        return hashlib.sha256(("summary-v2|" + content).encode()).hexdigest()

    def rebuild_relationship_summary(self) -> str:
        """Rebuild the relationship summary from all long-term memories.

        This distills all long-term and pinned memories into a concise
        paragraph that captures who the user is, what they care about,
        how they communicate, and what Koda should always know.

        The summary is deterministic (no LLM call) — it's assembled
        from structured memory data for speed and reliability.
        """
        # Gather source memories
        long_term = self._storage.search(MemoryQuery(
            tier=MemoryTier.LONG_TERM, limit=_SUMMARY_MAX_MEMORIES,
        ))
        pinned = self._storage.get_pinned()
        episodic = self._storage.search(MemoryQuery(
            tier=MemoryTier.EPISODIC, limit=50,
        ))

        # Merge and deduplicate
        seen: set[str] = set()
        all_memories: list[MemoryItem] = []
        for item in long_term + pinned + episodic:
            if item.id not in seen and not item.is_expired():
                seen.add(item.id)
                all_memories.append(item)

        if not all_memories:
            self._relationship_summary = ""
            self._relationship_hash = self._compute_memory_fingerprint()
            self._storage.save_relationship_summary("", self._relationship_hash)
            return ""

        # Organize by category
        by_category: dict[str, list[str]] = {}
        for item in all_memories:
            cat = item.category.value
            text = f"{item.key}: {item.content}" if item.key else item.content
            by_category.setdefault(cat, []).append(text)

        # Build structured summary
        parts: list[str] = []

        # Identity / facts
        if "fact" in by_category:
            facts = by_category["fact"][:10]
            parts.append("Key facts: " + "; ".join(facts))

        # Preferences
        if "preference" in by_category:
            prefs = by_category["preference"][:10]
            parts.append("Preferences: " + "; ".join(prefs))

        # Communication patterns
        if "pattern" in by_category:
            patterns = by_category["pattern"][:5]
            parts.append("Patterns: " + "; ".join(patterns))

        # Contacts / relationships
        if "contact" in by_category:
            contacts = by_category["contact"][:5]
            parts.append("Key people: " + "; ".join(contacts))

        # Active projects
        if "project" in by_category:
            projects = by_category["project"][:5]
            parts.append("Active projects: " + "; ".join(projects))

        # Notable events
        if "event" in by_category:
            events = by_category["event"][:5]
            parts.append("Notable events: " + "; ".join(events))

        # Health (if enabled)
        if "health" in by_category:
            health = by_category["health"][:3]
            parts.append("Health notes: " + "; ".join(health))

        # Koda's own reflections about this relationship
        if "reflection" in by_category:
            reflections = by_category["reflection"][:3]
            parts.append("Your reflections: " + "; ".join(reflections))

        summary = "\n".join(parts)

        # Truncate if needed
        if len(summary) > _SUMMARY_MAX_LENGTH:
            summary = summary[:_SUMMARY_MAX_LENGTH - 3] + "..."

        # Cache
        fingerprint = self._compute_memory_fingerprint()
        self._relationship_summary = summary
        self._relationship_hash = fingerprint
        self._storage.save_relationship_summary(summary, fingerprint)

        logger.info("Rebuilt relationship summary: %d chars from %d memories",
                     len(summary), len(all_memories))

        return summary

    def _ensure_summary_fresh(self) -> str:
        """Check if summary needs rebuilding and do so if needed.

        Returns the current (possibly refreshed) summary.
        """
        if not self._relationship_summary and self._storage.count() > 0:
            # No summary yet but we have memories — build one
            return self.rebuild_relationship_summary()

        # Check if memories have changed since last build
        current_hash = self._compute_memory_fingerprint()
        if current_hash != self._relationship_hash:
            return self.rebuild_relationship_summary()

        return self._relationship_summary

    # ------------------------------------------------------------------
    # Recall
    # ------------------------------------------------------------------

    def recall(
        self,
        text: Optional[str] = None,
        *,
        tier: Optional[MemoryTier] = None,
        category: Optional[MemoryCategory] = None,
        tags: Optional[list[str]] = None,
        limit: int = 50,
    ) -> list[MemoryItem]:
        """Search memories with optional filters."""
        query = MemoryQuery(
            text=text, tier=tier, category=category,
            tags=tags, limit=limit,
        )
        return self._storage.search(query)

    def get(self, item_id: str) -> Optional[MemoryItem]:
        """Get a specific memory by ID."""
        item = self._storage.get(item_id)
        if item is None or item.is_expired():
            return None
        item.touch()
        self._storage.save(item)
        return item

    # ------------------------------------------------------------------
    # Context building (hybrid keyword + semantic + pinned + summary)
    # ------------------------------------------------------------------

    def build_context(
        self,
        current_message: str = "",
        max_items: int = 30,
        max_tokens: int = 4000,
    ) -> ContextWindow:
        """Build a context window for the LLM.

        Three-layer injection:
        1. Relationship summary (stable, always present, ~150 tokens)
        2. Pinned memories (always present, user/Koda-selected core facts)
        3. Retrieved memories (dynamic, hybrid keyword + semantic search)

        Token budget increased from 2000→4000 to accommodate the richer
        context. Item limit increased from 20→30.
        """
        now = datetime.now(timezone.utc)

        # ---- Layer 1: Relationship summary ----
        summary = self._ensure_summary_fresh()
        summary_tokens = len(summary) // 4 if summary else 0

        # ---- Layer 2: Pinned memories (always included) ----
        pinned = self._storage.get_pinned()
        pinned_ids = {p.id for p in pinned}
        pinned_tokens = sum(
            (len(p.content) + len(p.key) + 20) // 4 for p in pinned
        )

        # ---- Layer 3: Retrieved memories (dynamic) ----
        # Reduce budget for retrieved items based on pinned + summary usage
        remaining_tokens = max_tokens - summary_tokens - pinned_tokens
        remaining_items = max_items - len(pinned)

        # Guard against negative budgets
        remaining_tokens = max(remaining_tokens, 500)
        remaining_items = max(remaining_items, 5)

        # Phase 1: Keyword/recency candidates
        candidates: list[MemoryItem] = []
        for tier in MemoryTier:
            items = self._storage.search(MemoryQuery(
                tier=tier, limit=remaining_items * 2,
            ))
            candidates.extend(items)

        if current_message:
            relevant = self._storage.search(MemoryQuery(
                text=current_message[:200], limit=remaining_items,
            ))
            seen_ids = {i.id for i in candidates}
            for item in relevant:
                if item.id not in seen_ids:
                    candidates.append(item)
                    seen_ids.add(item.id)

        # Phase 2: Semantic candidates
        semantic_scores: dict[str, float] = {}
        semantic_match_count = 0

        if current_message and self._embeddings and self._embeddings.index_size > 0:
            try:
                semantic_results = self._run_semantic_search(
                    current_message, top_k=remaining_items,
                )
                seen_ids_now = {i.id for i in candidates}
                for result in semantic_results:
                    semantic_scores[result.item_id] = result.similarity
                    if result.item_id not in seen_ids_now:
                        item = self._storage.get(result.item_id)
                        if item and not item.is_expired():
                            candidates.append(item)
                            seen_ids_now.add(item.id)
                            semantic_match_count += 1
            except Exception:
                logger.debug("Semantic search failed, falling back to keyword-only",
                             exc_info=True)

        # Phase 3: Deduplicate, filter expired, exclude pinned (they're separate)
        seen: set[str] = set()
        unique: list[MemoryItem] = []
        for item in candidates:
            if item.id not in seen and item.id not in pinned_ids and not item.is_expired():
                seen.add(item.id)
                unique.append(item)

        # Phase 4: Score + semantic boost
        msg_keywords = set(re.findall(r"\w{3,}", current_message.lower())) if current_message else set()
        _stopwords = {"the", "and", "for", "are", "but", "not", "you", "all",
                       "can", "her", "was", "one", "our", "out", "has", "have",
                       "from", "this", "that", "with", "they", "been", "said",
                       "each", "which", "their", "will", "way", "about", "many",
                       "then", "them", "would", "like", "into", "more", "some",
                       "what", "when", "who", "how", "where", "why", "does"}
        msg_keywords -= _stopwords

        scored: list[tuple[float, MemoryItem]] = []
        for item in unique:
            score = self._score_memory(item, msg_keywords, now)
            if item.id in semantic_scores:
                score += semantic_scores[item.id] * _SEMANTIC_BOOST
            scored.append((score, item))

        scored.sort(key=lambda x: x[0], reverse=True)

        # Phase 5: Select within remaining budget
        selected: list[MemoryItem] = []
        token_est = summary_tokens + pinned_tokens
        truncated = False

        for _score, item in scored:
            item_tokens = (len(item.content) + len(item.key) + 20) // 4
            if len(selected) >= remaining_items:
                truncated = True
                break
            if token_est + item_tokens > max_tokens:
                truncated = True
                continue
            selected.append(item)
            token_est += item_tokens

        return ContextWindow(
            items=selected,
            pinned_items=pinned,
            relationship_summary=summary,
            total_tokens_estimate=token_est,
            truncated=truncated,
            semantic_matches=semantic_match_count,
        )

    def _run_semantic_search(self, query: str, top_k: int = 20) -> list:
        """Run semantic search, handling both sync and async contexts."""
        if not self._embeddings:
            return []

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(
                    asyncio.run,
                    self._embeddings.search(query, top_k=top_k),
                )
                return future.result(timeout=10)
        else:
            return asyncio.run(self._embeddings.search(query, top_k=top_k))

    @staticmethod
    def _score_memory(
        item: MemoryItem,
        msg_keywords: set[str],
        now: datetime,
    ) -> float:
        """Score a memory item for context relevance."""
        tier_weights = {
            MemoryTier.LONG_TERM: 1.0,
            MemoryTier.EPISODIC: 0.9,
            MemoryTier.MEDIUM_TERM: 0.7,
            MemoryTier.SHORT_TERM: 0.5,
        }
        score = tier_weights.get(item.tier, 0.5)

        tier_halflife = {
            MemoryTier.SHORT_TERM: 0.25,
            MemoryTier.MEDIUM_TERM: 7.0,
            MemoryTier.LONG_TERM: 90.0,
            MemoryTier.EPISODIC: 30.0,
        }
        try:
            created = datetime.fromisoformat(item.created_at) if isinstance(item.created_at, str) else item.created_at
            if created.tzinfo is None:
                created = created.replace(tzinfo=timezone.utc)
            age_days = max((now - created).total_seconds() / 86400, 0.01)
            halflife = tier_halflife.get(item.tier, 30.0)
            decay = 0.5 ** (age_days / halflife)
        except (ValueError, TypeError):
            decay = 0.5
        score *= (0.3 + 0.7 * decay)

        if msg_keywords:
            item_words = set(re.findall(r"\w{3,}", item.content.lower()))
            if item.key:
                item_words |= set(re.findall(r"\w{3,}", item.key.lower()))
            overlap = len(msg_keywords & item_words)
            if overlap > 0:
                relevance = min(overlap / max(len(msg_keywords), 1), 1.0)
                score += relevance * 1.5

        access_count = getattr(item, "access_count", 0) or 0
        if access_count > 0:
            score += min(math.log(access_count + 1) * 0.1, 0.3)

        score *= max(item.confidence, 0.1)

        if item.category == MemoryCategory.PREFERENCE:
            score *= 1.2

        return score

    # ------------------------------------------------------------------
    # Privacy controls
    # ------------------------------------------------------------------

    def get_controls(self) -> list[CategoryControl]:
        return self._storage.get_category_controls()

    def set_control(self, control: CategoryControl) -> None:
        self._storage.set_category_control(control)
        if not control.enabled:
            deleted = self.forget_category(control.category)
            if deleted:
                logger.info("Disabled category %s — deleted %d memories",
                            control.category.value, deleted)
                self._invalidate_summary()

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------

    def cleanup_expired(self) -> int:
        all_items = self._storage.search(
            MemoryQuery(include_expired=True, limit=10000)
        )
        deleted = 0
        for item in all_items:
            if item.is_expired():
                self._storage.delete(item.id)
                if self._embeddings:
                    self._embeddings.remove_embedding(item.id)
                deleted += 1
        if deleted:
            self._invalidate_summary()
            logger.info("Cleaned up %d expired memories", deleted)
        return deleted

    def stats(self) -> MemoryStats:
        return self._storage.get_stats()

    def embedding_stats(self) -> EmbeddingStats:
        if not self._embeddings:
            return EmbeddingStats()
        raw = self._embeddings.get_stats()
        return EmbeddingStats(
            ready=raw["ready"],
            model=raw["model"],
            dims=raw["dims"],
            index_size=raw["index_size"],
            has_numpy=raw["has_numpy"],
            has_client=raw["has_client"],
        )

    def export_all(self) -> list[dict]:
        return self._storage.export_all()

    # ------------------------------------------------------------------
    # Auto-categorization
    # ------------------------------------------------------------------

    @staticmethod
    def _guess_category(content: str) -> MemoryCategory:
        lower = content.lower()
        words = set(re.findall(r"\w+", lower))

        best_cat = MemoryCategory.FACT
        best_score = 0

        for cat, hints in _CATEGORY_HINTS.items():
            score = len(words & hints)
            if score > best_score:
                best_score = score
                best_cat = cat

        return best_cat
