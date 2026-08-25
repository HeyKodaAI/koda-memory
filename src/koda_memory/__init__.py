"""Koda Memory System — persistent context across sessions.

Four-tier architecture:
  - Short-term: current session context (full detail)
  - Medium-term: past 30 days (summarized)
  - Long-term: key facts, preferences, learned patterns
  - Episodic: notable dated events

Retrieval: hybrid keyword + semantic search (v2).
  - Keyword: SQLite LIKE queries with multi-factor scoring
  - Semantic: OpenAI text-embedding-3-small vectors, cosine similarity
  - Combined: both pipelines feed into a unified scoring function

Privacy controls: category-level exclusions, one-click deletion,
full export and wipe.
"""
