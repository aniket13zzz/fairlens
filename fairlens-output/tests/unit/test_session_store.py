"""
Unit tests for MemorySessionStore.
Coverage: set/get/update/delete, TTL eviction, key not found.
"""
import time
import pytest
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from backend.session_store import MemorySessionStore


class TestMemorySessionStore:

    def test_set_and_get(self):
        store = MemorySessionStore()
        store.set("s1", {"df": "data", "analysis": None})
        result = store.get("s1")
        assert result is not None
        assert result["df"] == "data"

    def test_get_missing_returns_none(self):
        store = MemorySessionStore()
        assert store.get("nonexistent") is None

    def test_update_merges_patch(self):
        store = MemorySessionStore()
        store.set("s1", {"df": "data", "analysis": None})
        store.update("s1", {"analysis": {"severity": "SEVERE"}})
        result = store.get("s1")
        assert result["analysis"]["severity"] == "SEVERE"
        assert result["df"] == "data"

    def test_update_missing_raises(self):
        store = MemorySessionStore()
        with pytest.raises(KeyError):
            store.update("ghost", {"x": 1})

    def test_delete_removes_session(self):
        store = MemorySessionStore()
        store.set("s1", {"df": "data"})
        store.delete("s1")
        assert store.get("s1") is None

    def test_ttl_eviction(self):
        store = MemorySessionStore()
        # Manually set expiry in the past
        store.set("s1", {"df": "data"})
        store._expiry["s1"] = time.monotonic() - 1  # already expired
        assert store.get("s1") is None

    def test_evict_only_expired(self):
        store = MemorySessionStore()
        store.set("fresh", {"df": "data"})
        store.set("stale", {"df": "old"})
        store._expiry["stale"] = time.monotonic() - 1
        store._evict_expired()
        assert store.get("fresh") is not None
        assert store.get("stale") is None

    def test_multiple_sessions_isolated(self):
        store = MemorySessionStore()
        store.set("a", {"val": 1})
        store.set("b", {"val": 2})
        store.update("a", {"val": 99})
        assert store.get("b")["val"] == 2
        assert store.get("a")["val"] == 99
