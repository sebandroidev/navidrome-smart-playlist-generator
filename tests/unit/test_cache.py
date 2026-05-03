"""
Unit tests for ingestion/cache.py

Covers: get, set, invalidate, clear, TTL expiry, thread-safety basics.
"""
import time
import pytest
import threading

import ingestion.cache as cache


@pytest.fixture(autouse=True)
def _clean_cache():
    """Wipe the cache store before and after every test."""
    cache.clear()
    yield
    cache.clear()


# ---------------------------------------------------------------------------
# set / get — basic round-trip
# ---------------------------------------------------------------------------

class TestCacheSetGet:
    def test_get_missing_key_returns_none(self):
        assert cache.get("nonexistent", 60) is None

    def test_set_then_get_returns_value(self):
        cache.set("key1", "hello")
        assert cache.get("key1", 60) == "hello"

    def test_get_with_long_ttl_returns_value(self):
        cache.set("k", [1, 2, 3])
        assert cache.get("k", 9999) == [1, 2, 3]

    def test_stores_none_value(self):
        # None is a valid cached value — must not be confused with a cache miss
        cache.set("nullkey", None)
        # The entry exists but value is None; get still returns None (same as miss)
        # Implementation stores (None, ts) and returns the value — which is None.
        result = cache.get("nullkey", 60)
        assert result is None

    def test_stores_dict_value(self):
        data = {"a": 1, "b": [2, 3]}
        cache.set("dict_key", data)
        assert cache.get("dict_key", 300) == data

    def test_stores_integer_value(self):
        cache.set("int_key", 42)
        assert cache.get("int_key", 60) == 42

    def test_stores_list_value(self):
        lst = [{"id": "x"}, {"id": "y"}]
        cache.set("list_key", lst)
        assert cache.get("list_key", 60) == lst

    def test_overwrite_key_returns_new_value(self):
        cache.set("ow", "first")
        cache.set("ow", "second")
        assert cache.get("ow", 60) == "second"


# ---------------------------------------------------------------------------
# TTL expiry
# ---------------------------------------------------------------------------

class TestCacheTTL:
    def test_expired_entry_returns_none(self):
        cache.set("ttl_key", "value")
        # Use a TTL of 0 seconds — the entry should already be expired
        result = cache.get("ttl_key", ttl_seconds=0)
        assert result is None

    def test_not_yet_expired_entry_returns_value(self):
        cache.set("fresh", "still-alive")
        # Large TTL — should not be expired
        assert cache.get("fresh", ttl_seconds=3600) == "still-alive"

    def test_expired_entry_removed_from_store(self):
        cache.set("evict", "val")
        # Expired immediately
        cache.get("evict", ttl_seconds=0)
        # A subsequent call with a long TTL also returns None (was already evicted)
        assert cache.get("evict", ttl_seconds=3600) is None

    def test_very_small_ttl_expires_quickly(self):
        cache.set("tiny_ttl", "quick")
        time.sleep(0.05)
        # 0.04s TTL — must be expired after 50ms sleep
        result = cache.get("tiny_ttl", ttl_seconds=0.04)
        assert result is None

    def test_large_ttl_not_expired_after_short_sleep(self):
        cache.set("big_ttl", "persists")
        time.sleep(0.01)
        assert cache.get("big_ttl", ttl_seconds=600) == "persists"


# ---------------------------------------------------------------------------
# invalidate
# ---------------------------------------------------------------------------

class TestCacheInvalidate:
    def test_invalidate_removes_key(self):
        cache.set("rm", "gone")
        cache.invalidate("rm")
        assert cache.get("rm", 60) is None

    def test_invalidate_nonexistent_key_no_error(self):
        cache.invalidate("does_not_exist")  # Should not raise

    def test_invalidate_one_key_leaves_others(self):
        cache.set("a", 1)
        cache.set("b", 2)
        cache.invalidate("a")
        assert cache.get("a", 60) is None
        assert cache.get("b", 60) == 2

    def test_invalidate_then_reset_works(self):
        cache.set("reuse", "old")
        cache.invalidate("reuse")
        cache.set("reuse", "new")
        assert cache.get("reuse", 60) == "new"


# ---------------------------------------------------------------------------
# clear
# ---------------------------------------------------------------------------

class TestCacheClear:
    def test_clear_removes_all_keys(self):
        cache.set("x", 1)
        cache.set("y", 2)
        cache.set("z", 3)
        cache.clear()
        assert cache.get("x", 60) is None
        assert cache.get("y", 60) is None
        assert cache.get("z", 60) is None

    def test_clear_empty_store_no_error(self):
        cache.clear()  # Already empty from autouse fixture — should not raise

    def test_clear_then_set_works(self):
        cache.set("before", "old")
        cache.clear()
        cache.set("after", "new")
        assert cache.get("after", 60) == "new"
        assert cache.get("before", 60) is None


# ---------------------------------------------------------------------------
# Thread safety smoke test
# ---------------------------------------------------------------------------

class TestCacheThreadSafety:
    def test_concurrent_writes_do_not_raise(self):
        errors = []

        def writer(n):
            try:
                for i in range(50):
                    cache.set(f"thread{n}_key{i}", i)
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=writer, args=(n,)) for n in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []

    def test_concurrent_reads_writes_do_not_raise(self):
        errors = []
        cache.set("shared", "initial")

        def reader():
            try:
                for _ in range(100):
                    cache.get("shared", 60)
            except Exception as exc:
                errors.append(exc)

        def writer():
            try:
                for i in range(100):
                    cache.set("shared", i)
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=reader) for _ in range(3)]
        threads += [threading.Thread(target=writer) for _ in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []
