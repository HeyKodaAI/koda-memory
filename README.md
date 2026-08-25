# koda-memory

Tiered persistent memory for AI agents: short/medium/long-term + episodic tiers, hybrid keyword + semantic retrieval, category-level privacy controls, and a secret-leak detector.

## What & why

An agent without memory re-learns its user every session. `koda-memory` gives an agent SQLite-backed persistent memory organized into tiers with automatic expiry, retrieves the most relevant memories for each incoming message, and assembles them into an LLM-ready prompt section. It also ships a `SecretDetector`, because agent memory is a common credential-leak vector: users paste passwords and API keys into chat, and a naive memory system stores them in plain text forever. The detector catches credential-shaped content so it can be routed to encrypted storage and redacted before anything reaches memory.

Extracted from [Koda AI](https://github.com/HeyKodaAI) ("the AI agent that never lies"), where it was the agent's cross-session memory. Part of the Koda project by Mike Wandzilak / Wandzilak Web Design.

## Features

- **Four memory tiers** — `short_term` (expires in 24h), `medium_term` (30 days), `long_term` and `episodic` (never expire). Per-category expiry overrides supported.
- **Eleven categories** — preference, project, contact, task, pattern, fact, conversation, event, health, financial, reflection — with keyword-based auto-categorization when none is given.
- **Privacy controls** — disable a category and new memories in it are refused while existing ones are deleted; set `auto_expire_days` per category; full export and one-call wipe.
- **Hybrid retrieval** — keyword search (SQLite, zero-config default) plus optional semantic search via vector embeddings; both feed a multi-factor scorer (tier weight, age decay with per-tier half-life, keyword overlap, access count, confidence).
- **Context windows** — `build_context()` assembles three layers under a token budget: a relationship summary (deterministic condensation of long-term memories, no LLM call), pinned memories (always included), and dynamically retrieved memories.
- **Secret detection** — regex patterns for known credential formats (Anthropic/OpenAI/GitHub/AWS/Slack/Discord keys, SSH private keys, generic long tokens) plus natural-language patterns ("my netflix password is ..."), with confidence scoring and a `redact()` helper. The detection result's `repr` never prints the secret value.
- **FastAPI routes** — a ready-made `APIRouter` exposing the full memory API over HTTP.

## Install

Not yet on PyPI. Requires Python 3.11+.

```bash
pip install git+https://github.com/HeyKodaAI/koda-memory.git
```

## Quickstart

Keyword search is the zero-config default — no API keys, no extra dependencies.

```python
from koda_memory.manager import MemoryManager
from koda_memory.models import MemoryCategory, MemoryTier
from koda_memory.secret_detector import SecretDetector
from koda_memory.storage import MemoryStorage

# In-memory SQLite for the demo; use a file path for persistence.
storage = MemoryStorage(":memory:")
manager = MemoryManager(storage)

manager.remember(
    "Prefers dark mode and concise answers",
    key="ui_preferences",
    category=MemoryCategory.PREFERENCE,
)
manager.remember(
    "Working on the Q3 marketing site redesign",
    key="current_project",
    tier=MemoryTier.MEDIUM_TERM,
    category=MemoryCategory.PROJECT,
)

# Keyword search (zero-config default — no API key needed)
for item in manager.recall("dark mode"):
    print(f"recalled: {item.key} -> {item.content}")

# Build an LLM-ready context window for the current message
ctx = manager.build_context("what theme should I use?")
print("---")
print(ctx.to_prompt_section())
print("---")

# Secret detection: block credentials from entering plain-text memory
detector = SecretDetector()
result = detector.scan("my github token is ghp_" + "x" * 36)
print(result)  # repr never prints the secret value
if result.detected:
    print(detector.redact(result.original_message, result))
```

Output:

```text
recalled: ui_preferences -> Prefers dark mode and concise answers
---
## Who this person is to you:

Preferences: ui_preferences: Prefers dark mode and concise answers

## What you remember about the user:

- [preference] **ui_preferences**: Prefers dark mode and concise answers
- [project] **current_project**: Working on the Q3 marketing site redesign
---
SecretDetection(detected=True, confidence=0.95, type='token', service='github')
my github token is [CREDENTIAL_STORED_IN_VAULT]
```

Note: the top-level `koda_memory` package holds only documentation; import from the submodules as shown above.

## Semantic search (optional)

`EmbeddingEngine` (in `koda_memory.embeddings`) adds retrieval by meaning. It generates embeddings through the OpenAI client library — either OpenAI's `text-embedding-3-small` (the default, requires an OpenAI API key) or a local Ollama-compatible endpoint (`nomic-embed-text` and similar models, no API cost). Vectors are stored as float32 BLOBs in SQLite and searched via an in-memory numpy cosine-similarity index. It requires `numpy` and `openai`, which are not declared dependencies — install them yourself if you want it. Everything degrades gracefully to keyword-only search when the engine is absent, unconfigured, or failing.

```python
engine = EmbeddingEngine(db_path="memory.db")
await engine.init(api_key="sk-...")        # or model="nomic-embed-text" for local Ollama
manager.set_embedding_engine(engine)        # build_context() now blends both pipelines
```

## API overview

- `koda_memory.manager.MemoryManager` — the high-level API: `remember`, `recall`, `get`, `update`, `forget`, `forget_category`, `wipe`, `pin_memory`/`unpin_memory`/`get_pinned`, `build_context`, `rebuild_relationship_summary`, `get_controls`/`set_control`, `cleanup_expired`, `stats`, `export_all`, `add_on_remember_hook`.
- `koda_memory.storage.MemoryStorage` — SQLite persistence (WAL mode) with schema migration; `search(MemoryQuery)` is the low-level query path.
- `koda_memory.models` — `MemoryItem`, `MemoryTier`, `MemoryCategory`, `MemoryQuery`, `CategoryControl`, `MemoryStats`, `ContextWindow` (with `to_prompt_section()`), `EmbeddingStats`.
- `koda_memory.secret_detector.SecretDetector` — `scan(message) -> SecretDetection`, `redact(message, detection)`.
- `koda_memory.embeddings.EmbeddingEngine` — `init`, `embed_memory`, `embed_memory_batch`, `search`, `search_by_vector`, `backfill_from_storage`, `cleanup_orphaned`.
- `koda_memory.seeds` — `get_seed_memories()` / `load_seed_memories(manager)`: example identity seeds loaded once at first boot and source-tagged `inception` so they can never be confused with user memories. The shipped set is Koda's public origin story — replace it with your own agent's foundation.
- `koda_memory.routes` — FastAPI `APIRouter` at `/api/v1/memory`. Call `init_memory_routes(manager)` at startup, then include `routes.router` in your app.

| Method | Path | Purpose |
|--------|------|---------|
| GET / POST | `/api/v1/memory` | Search / store memories |
| GET | `/api/v1/memory/stats` | Statistics incl. embedding engine state |
| GET | `/api/v1/memory/export` | Export all memories |
| GET | `/api/v1/memory/context` | Build a context window |
| GET | `/api/v1/memory/pinned` | List pinned memories |
| POST | `/api/v1/memory/{id}/pin` · `/{id}/unpin` | Pin / unpin |
| GET / POST | `/api/v1/memory/summary` · `/summary/rebuild` | Relationship summary |
| GET / PUT / DELETE | `/api/v1/memory/{id}` | Get / update / delete one memory |
| DELETE | `/api/v1/memory/category/{cat}` | Delete a whole category |
| GET / PUT | `/api/v1/memory/controls` · `/controls/{cat}` | Privacy controls |
| POST | `/api/v1/memory/cleanup` | Expiry cleanup |
| POST | `/api/v1/memory/embeddings/backfill` | Backfill embeddings |
| POST | `/api/v1/memory/wipe` | Delete everything |

## Testing

```bash
pip install -e . pytest pytest-asyncio
pytest
```

48 tests.

## License

MIT.
