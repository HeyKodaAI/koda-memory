"""Memory system API routes.

Endpoints:
    GET    /api/v1/memory                    List/search memories
    POST   /api/v1/memory                    Store a memory
    GET    /api/v1/memory/stats              Memory statistics
    GET    /api/v1/memory/export             Export all memories
    GET    /api/v1/memory/context            Build context window
    GET    /api/v1/memory/pinned             List pinned memories
    POST   /api/v1/memory/{id}/pin           Pin a memory
    POST   /api/v1/memory/{id}/unpin         Unpin a memory
    GET    /api/v1/memory/summary            Get relationship summary
    POST   /api/v1/memory/summary/rebuild    Force rebuild relationship summary
    GET    /api/v1/memory/{id}               Get a specific memory
    PUT    /api/v1/memory/{id}               Update a memory
    DELETE /api/v1/memory/{id}               Delete a memory
    DELETE /api/v1/memory/category/{cat}     Delete all in category
    POST   /api/v1/memory/wipe              Wipe all memories
    GET    /api/v1/memory/controls           Get privacy controls
    PUT    /api/v1/memory/controls/{cat}     Set privacy control
    POST   /api/v1/memory/cleanup            Run expiry cleanup
    POST   /api/v1/memory/embeddings/backfill  Backfill embeddings
"""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from koda_memory.manager import MemoryManager
from koda_memory.models import (
    CategoryControl,
    MemoryCategory,
    MemoryTier,
)

router = APIRouter(prefix="/api/v1/memory", tags=["memory"])

_manager: Optional[MemoryManager] = None


def init_memory_routes(manager: MemoryManager) -> None:
    global _manager
    _manager = manager


def _get_manager() -> MemoryManager:
    if _manager is None:
        raise RuntimeError("Memory manager not initialized")
    return _manager


# --- Request/Response models ---

class MemoryCreateRequest(BaseModel):
    content: str = Field(..., min_length=1, max_length=50000)
    key: str = ""
    tier: str = "long_term"
    category: Optional[str] = None
    source: str = ""
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    tags: list[str] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)
    importance: Optional[float] = Field(default=None, ge=0.0, le=1.0)  # Accepted but not used; kept for compatibility


class MemoryUpdateRequest(BaseModel):
    content: str


class CategoryControlRequest(BaseModel):
    enabled: bool = True
    auto_expire_days: Optional[int] = None


# --- Endpoints ---

@router.get("")
async def list_memories(
    text: Optional[str] = None,
    tier: Optional[str] = None,
    category: Optional[str] = None,
    tag: Optional[str] = None,
    limit: int = 50,
) -> dict:
    """List or search memories."""
    mgr = _get_manager()

    tier_enum = MemoryTier(tier) if tier else None
    cat_enum = MemoryCategory(category) if category else None
    tags = [tag] if tag else None

    items = mgr.recall(
        text=text, tier=tier_enum, category=cat_enum,
        tags=tags, limit=limit,
    )
    return {
        "memories": [_item_to_dict(i) for i in items],
        "count": len(items),
    }


@router.post("", status_code=status.HTTP_201_CREATED)
async def store_memory(req: MemoryCreateRequest) -> dict:
    """Store a new memory."""
    mgr = _get_manager()

    try:
        tier = MemoryTier(req.tier)
    except ValueError:
        valid_tiers = [t.value for t in MemoryTier]
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid tier '{req.tier}'. Must be one of: {valid_tiers}",
        )

    category = None
    if req.category:
        try:
            category = MemoryCategory(req.category)
        except ValueError:
            # Try to fuzzy-match (e.g. "preferences" → "preference")
            for mc in MemoryCategory:
                if mc.value.startswith(req.category.rstrip("s")):
                    category = mc
                    break

    item = mgr.remember(
        content=req.content,
        key=req.key,
        tier=tier,
        category=category,
        source=req.source,
        confidence=req.confidence,
        tags=req.tags,
        metadata=req.metadata,
    )

    if item is None:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Category is disabled by privacy controls",
        )

    return {"id": item.id, "message": "Memory stored", "item": _item_to_dict(item)}


@router.get("/stats")
async def memory_stats() -> dict:
    """Get memory statistics including embedding engine info."""
    mgr = _get_manager()
    stats = mgr.stats()
    emb_stats = mgr.embedding_stats()
    return {
        "total_items": stats.total_items,
        "by_tier": stats.by_tier,
        "by_category": stats.by_category,
        "oldest_item": stats.oldest_item,
        "newest_item": stats.newest_item,
        "embeddings": {
            "ready": emb_stats.ready,
            "model": emb_stats.model,
            "dims": emb_stats.dims,
            "index_size": emb_stats.index_size,
            "has_numpy": emb_stats.has_numpy,
            "has_client": emb_stats.has_client,
        },
    }


@router.get("/export")
async def export_memories() -> dict:
    """Export all memories."""
    mgr = _get_manager()
    data = mgr.export_all()
    return {"memories": data, "count": len(data)}


@router.get("/context")
async def build_context(
    message: str = "",
    max_items: int = 20,
    max_tokens: int = 2000,
) -> dict:
    """Build a context window for LLM use."""
    mgr = _get_manager()
    ctx = mgr.build_context(
        current_message=message,
        max_items=max_items,
        max_tokens=max_tokens,
    )
    return {
        "items": [_item_to_dict(i) for i in ctx.items],
        "pinned_items": [_item_to_dict(i) for i in ctx.pinned_items],
        "relationship_summary": ctx.relationship_summary,
        "count": len(ctx.items),
        "pinned_count": len(ctx.pinned_items),
        "total_tokens_estimate": ctx.total_tokens_estimate,
        "truncated": ctx.truncated,
        "semantic_matches": ctx.semantic_matches,
        "prompt_section": ctx.to_prompt_section(),
    }


@router.get("/controls")
async def get_controls() -> dict:
    """Get privacy controls for all categories."""
    mgr = _get_manager()
    controls = mgr.get_controls()
    return {
        "controls": [
            {
                "category": c.category.value,
                "enabled": c.enabled,
                "auto_expire_days": c.auto_expire_days,
            }
            for c in controls
        ],
    }


@router.put("/controls/{category}")
async def set_control(category: str, req: CategoryControlRequest) -> dict:
    """Set privacy control for a category."""
    mgr = _get_manager()
    try:
        cat = MemoryCategory(category)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid category: {category}",
        )

    control = CategoryControl(
        category=cat,
        enabled=req.enabled,
        auto_expire_days=req.auto_expire_days,
    )
    mgr.set_control(control)
    return {"message": f"Control updated for {category}", "enabled": req.enabled}


@router.post("/cleanup")
async def cleanup_expired() -> dict:
    """Run expiry cleanup."""
    mgr = _get_manager()
    deleted = mgr.cleanup_expired()
    return {"deleted": deleted}


@router.post("/embeddings/backfill")
async def backfill_embeddings() -> dict:
    """Backfill embeddings for memories that don't have one yet."""
    mgr = _get_manager()
    if not mgr.embeddings or not mgr.embeddings.ready:
        return {
            "success": False,
            "message": "Embedding engine not available",
            "total": 0, "embedded": 0, "skipped": 0, "failed": 0,
        }

    result = await mgr.embeddings.backfill_from_storage(mgr.storage)
    return {"success": True, **result}


@router.post("/wipe")
async def wipe_all() -> dict:
    """Delete ALL memories. Use with extreme caution."""
    mgr = _get_manager()
    deleted = mgr.wipe()
    return {"deleted": deleted, "message": "All memories wiped"}


# --- Pinned memories ---

@router.get("/pinned")
async def list_pinned() -> dict:
    """List all pinned memories."""
    mgr = _get_manager()
    pinned = mgr.get_pinned()
    return {
        "memories": [_item_to_dict(i) for i in pinned],
        "count": len(pinned),
    }


@router.post("/{item_id}/pin")
async def pin_memory(item_id: str) -> dict:
    """Pin a memory so it's always included in context."""
    mgr = _get_manager()
    success = mgr.pin_memory(item_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Memory {item_id} not found",
        )
    return {"message": "Memory pinned", "id": item_id, "pinned": True}


@router.post("/{item_id}/unpin")
async def unpin_memory(item_id: str) -> dict:
    """Unpin a memory."""
    mgr = _get_manager()
    success = mgr.unpin_memory(item_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Memory {item_id} not found",
        )
    return {"message": "Memory unpinned", "id": item_id, "pinned": False}


# --- Relationship summary ---

@router.get("/summary")
async def get_summary() -> dict:
    """Get the current relationship summary."""
    mgr = _get_manager()
    return {
        "summary": mgr.relationship_summary,
        "length": len(mgr.relationship_summary),
    }


@router.post("/summary/rebuild")
async def rebuild_summary() -> dict:
    """Force rebuild the relationship summary from current memories."""
    mgr = _get_manager()
    summary = mgr.rebuild_relationship_summary()
    return {
        "summary": summary,
        "length": len(summary),
        "message": "Relationship summary rebuilt",
    }


@router.get("/{item_id}")
async def get_memory(item_id: str) -> dict:
    """Get a specific memory."""
    mgr = _get_manager()
    item = mgr.get(item_id)
    if not item:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Memory {item_id} not found",
        )
    return _item_to_dict(item)


@router.put("/{item_id}")
async def update_memory(item_id: str, req: MemoryUpdateRequest) -> dict:
    """Update a memory's content."""
    mgr = _get_manager()
    success = mgr.update(item_id, req.content)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Memory {item_id} not found",
        )
    return {"message": "Memory updated", "id": item_id}


@router.delete("/{item_id}")
async def delete_memory(item_id: str) -> dict:
    """Delete a memory."""
    mgr = _get_manager()
    success = mgr.forget(item_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Memory {item_id} not found",
        )
    return {"message": "Memory deleted", "id": item_id}


@router.delete("/category/{category}")
async def delete_category(category: str) -> dict:
    """Delete all memories in a category."""
    mgr = _get_manager()
    try:
        cat = MemoryCategory(category)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid category: {category}",
        )
    deleted = mgr.forget_category(cat)
    return {"deleted": deleted, "category": category}


# --- Helpers ---

def _item_to_dict(item) -> dict:
    return {
        "id": item.id,
        "tier": item.tier.value,
        "category": item.category.value,
        "key": item.key,
        "content": item.content,
        "source": item.source,
        "confidence": item.confidence,
        "tags": item.tags,
        "metadata": item.metadata,
        "created_at": item.created_at,
        "updated_at": item.updated_at,
        "expires_at": item.expires_at,
        "access_count": item.access_count,
        "pinned": getattr(item, "pinned", False),
    }
