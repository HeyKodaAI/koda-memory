import pytest
from koda_memory.manager import MemoryManager
from koda_memory.storage import MemoryStorage
from koda_memory.models import MemoryTier, MemoryCategory, CategoryControl
from koda_memory.secret_detector import SecretDetector


@pytest.mark.parametrize("operation", ["forget", "category", "disable"])
def test_deleted_episodic_summary_stays_deleted_after_restart(tmp_path, operation):
    path = str(tmp_path / "memory.db")
    manager = MemoryManager(MemoryStorage(path))
    item = manager.remember("synthetic-private-health", tier=MemoryTier.EPISODIC,
                            category=MemoryCategory.HEALTH)
    assert "synthetic-private-health" in manager.build_context().to_prompt_section()
    if operation == "forget":
        manager.forget(item.id)
    elif operation == "category":
        manager.forget_category(MemoryCategory.HEALTH)
    else:
        manager.set_control(CategoryControl(category=MemoryCategory.HEALTH, enabled=False))
    assert "synthetic-private-health" not in manager.relationship_summary
    assert "synthetic-private-health" not in manager.storage.get_relationship_summary()
    restarted = MemoryManager(MemoryStorage(path))
    assert "synthetic-private-health" not in restarted.build_context().to_prompt_section()


def test_episodic_update_refreshes_summary():
    manager = MemoryManager(MemoryStorage(":memory:"))
    item = manager.remember("old event", tier=MemoryTier.EPISODIC, category=MemoryCategory.EVENT)
    manager.build_context()
    manager.update(item.id, "new event")
    text = manager.build_context().to_prompt_section()
    assert "new event" in text and "old event" not in text


def test_expired_pinned_and_direct_retrieval_are_filtered():
    manager = MemoryManager(MemoryStorage(":memory:"))
    item = manager.remember("expired-fact", pinned=True)
    manager.build_context()
    item.expires_at = "2000-01-01T00:00:00+00:00"
    manager.storage.save(item)
    assert manager.get(item.id) is None
    assert manager.get_pinned() == []
    assert "expired-fact" not in manager.build_context().to_prompt_section()
    # Administrative storage/export still retains the row until cleanup.
    assert manager.storage.get(item.id) is not None


def test_episodic_expiry_refreshes_summary_without_cleanup():
    manager = MemoryManager(MemoryStorage(":memory:"))
    item = manager.remember("expired-event", tier=MemoryTier.EPISODIC, category=MemoryCategory.EVENT)
    manager.build_context()
    item.expires_at = "2000-01-01T00:00:00+00:00"
    manager.storage.save(item)
    assert "expired-event" not in manager.build_context().to_prompt_section()


def test_all_same_format_secrets_are_redacted_and_available_for_vault():
    values = ["ghp_" + "A1" * 18, "ghp_" + "B2" * 18]
    message = " and ".join(values)
    detector = SecretDetector()
    assert {d.secret_value for d in detector.scan_all(message)} == set(values)
    for output in [detector.redact_all(message), detector.redact(message, detector.scan(message))]:
        assert all(value not in output for value in values)
        assert "STORED" not in output


def test_mixed_formats_and_keyword_password():
    values = ["ghp_" + "A1" * 18, "xoxb-" + "B2" * 10, "test-password"]
    message = values[0] + " " + values[1] + "\nmy netflix password is " + values[2]
    detector = SecretDetector()
    assert all(value not in detector.redact_all(message) for value in values)
    assert all(any(value == d.secret_value for d in detector.scan_all(message)) for value in values)


def test_entire_private_key_block_is_redacted():
    key = "-----BEGIN OPENSSH PRIVATE KEY-----\n" + "Ab12" * 25 + "\n-----END OPENSSH PRIVATE KEY-----"
    detector = SecretDetector()
    assert detector.redact_all(key) == "[REDACTED]"
    assert detector.scan_all(key)[0].secret_value == key


def test_plaintext_and_high_confidence_threshold():
    detector = SecretDetector(min_confidence=0.99)
    assert detector.scan_all("my netflix password is example") == []
    assert SecretDetector().redact_all("The weather is pleasant") == "The weather is pleasant"


def test_old_summary_cache_is_revalidated_on_upgrade(tmp_path):
    import hashlib
    path = str(tmp_path / "memory.db")
    storage = MemoryStorage(path)
    storage.save_relationship_summary("deleted legacy health text", hashlib.md5(b"").hexdigest())
    manager = MemoryManager(storage)
    assert manager.relationship_summary == ""
    assert storage.get_relationship_summary() == ""
