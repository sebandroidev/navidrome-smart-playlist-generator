import time
import threading
from typing import Any, Optional

_store: dict[str, tuple[Any, float]] = {}
_lock = threading.Lock()


def get(key: str, ttl_seconds: float) -> Optional[Any]:
    with _lock:
        entry = _store.get(key)
        if entry is None:
            return None
        value, ts = entry
        if time.monotonic() - ts > ttl_seconds:
            del _store[key]
            return None
        return value


def set(key: str, value: Any):
    with _lock:
        _store[key] = (value, time.monotonic())


def invalidate(key: str):
    with _lock:
        _store.pop(key, None)


def clear():
    with _lock:
        _store.clear()
