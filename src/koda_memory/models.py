"""Memory system data models."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional
import uuid


class MemoryTier(str, Enum):
    """How long a memory persists and at what detail level."""
    SHORT_TERM = "short_term"     # Current session, full detail
    MEDIUM_TERM = "medium_term"   # Past 30 days, summarized
    LONG_TERM = "long_term"       # Permanent key facts / preferences
    EPISODIC = "episodic"         # Notable dated events


class MemoryCategory(str, Enum):
    """Semantic category for privacy controls."""
    PREFERENCE = "preference"         # User preferences, style, timezone
    PROJECT = "project"               # Active projects, deadlines
    CONTACT = "contact"               # Key people, relationships
    TASK = "task"                     # Recurring tasks, completed work
    PATTERN = "pattern"               # Learned behavioral patterns
    FACT = "fact"                     # Miscellaneous facts about user
    CONVERSATION = "conversation"     # Summarized conversation history
    EVENT = "event"                   # Notable events (episodic)
    HEALTH = "health"                 # Health-related (opt-out target)
    FINANCIAL = "financial"           # Financial info (opt-out target)
    REFLECTION = "reflection"         # Koda's own thoughts and self-awareness


@dataclass
class MemoryItem:
    """A single memory entry."""
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    tier: MemoryTier = MemoryTier.LONG_TERM
    category: MemoryCategory = MemoryCategory.FACT
    key: str = ""                   # Short label, e.g. "timezone"
    content: str = ""               # The actual memory content
    source: str = ""                # Where this came from (conversation id, manual, etc.)
    confidence: float = 1.0         # 0.0-1.0, how certain we are
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    updated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    expires_at: Optional[str] = None  # Auto-expire for short/medium term
    access_count: int = 0
    pinned: bool = False             # Always included in context window

    def is_expired(self) -> bool:
        """Check if this memory has expired."""
        if self.expires_at is None:
            return False
        expiry = datetime.fromisoformat(self.expires_at)
        return datetime.now(timezone.utc) >= expiry

    def touch(self) -> None:
        """Record an access and update timestamp."""
        self.access_count += 1
        self.updated_at = datetime.now(timezone.utc).isoformat()


@dataclass
class MemoryQuery:
    """Query parameters for searching memory."""
    text: Optional[str] = None        # Free-text search
    tier: Optional[MemoryTier] = None
    category: Optional[MemoryCategory] = None
    tags: Optional[list[str]] = None
    min_confidence: float = 0.0
    limit: int = 50
    include_expired: bool = False


@dataclass
class CategoryControl:
    """Privacy control for a memory category."""
    category: MemoryCategory
    enabled: bool = True              # False = don't remember this category
    auto_expire_days: Optional[int] = None  # Auto-expire after N days


@dataclass
class MemoryStats:
    """Summary statistics about stored memories."""
    total_items: int = 0
    by_tier: dict[str, int] = field(default_factory=dict)
    by_category: dict[str, int] = field(default_factory=dict)
    oldest_item: Optional[str] = None
    newest_item: Optional[str] = None


@dataclass
class ContextWindow:
    """Assembled context for the LLM from memory.

    The memory manager builds this before each request,
    selecting the most relevant memories for the current conversation.
    """
    items: list[MemoryItem] = field(default_factory=list)
    pinned_items: list[MemoryItem] = field(default_factory=list)  # Always-included memories
    relationship_summary: str = ""  # Condensed portrait of the user
    total_tokens_estimate: int = 0
    truncated: bool = False
    semantic_matches: int = 0  # How many items came from semantic search

    def to_prompt_section(self) -> str:
        """Format memories as a prompt section for the LLM.

        Structure:
        1. Relationship summary (stable, always present)
        2. Pinned memories (core facts, always present)
        3. Retrieved memories (dynamic, message-dependent)
        """
        sections: list[str] = []

        # 1. Relationship summary — the stable emotional anchor
        if self.relationship_summary:
            sections.append(
                "## Who this person is to you:\n\n" + self.relationship_summary
            )

        # 2. Pinned memories — always present
        if self.pinned_items:
            lines = ["## Core memories (always available):\n"]
            for item in self.pinned_items:
                prefix = f"[{item.category.value}]"
                if item.key:
                    lines.append(f"- {prefix} **{item.key}**: {item.content}")
                else:
                    lines.append(f"- {prefix} {item.content}")
            sections.append("\n".join(lines))

        # 3. Retrieved memories — contextual
        if self.items:
            lines = ["## What you remember about the user:\n"]
            for item in self.items:
                prefix = f"[{item.category.value}]"
                if item.key:
                    lines.append(f"- {prefix} **{item.key}**: {item.content}")
                else:
                    lines.append(f"- {prefix} {item.content}")
            sections.append("\n".join(lines))

        return "\n\n".join(sections) if sections else ""


@dataclass
class EmbeddingStats:
    """Statistics about the embedding engine."""
    ready: bool = False
    model: str = ""
    dims: int = 0
    index_size: int = 0
    has_numpy: bool = False
    has_client: bool = False
