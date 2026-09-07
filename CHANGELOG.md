# Changelog

## 0.1.1 - 2026-09-07

- Summary deletion clears both memory and SQLite caches. Episodic memories participate in the fingerprint; older cache fingerprints are rebuilt on startup, and direct summary reads refresh expiry. (Review 3)
- Expired memories are excluded from normal direct and pinned retrieval. Administrative storage/export can still inspect retained records until cleanup. (Review 4)
- Added `scan_all()` and `redact_all()`; the existing `redact()` also removes other recognized credentials in the message. Complete private-key blocks are handled as one detection. (Review 5)

### Upgrade

Redaction now uses `[REDACTED]`, rather than claiming a credential was stored. Use `scan_all(message)` and store each returned credential in a vault before discarding the original. Only then pass the redacted message to memory. The single-result `scan()` remains available but cannot enumerate multiple credentials. Detection is heuristic, not proof that arbitrary input is secret-free. `remember()` itself does not automatically call the detector.

### Validation

59 tests pass, including `tests/test_review_regressions.py`; main README quickstart checked. Tests use synthetic data and isolated databases. No live provider calls or service actions were used.
