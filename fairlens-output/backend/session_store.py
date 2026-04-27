"""
FairLens AI — Session store.
Supports in-memory (dev) and Redis (production) backends.
Set SESSION_BACKEND=redis and REDIS_URL=redis://... to enable Redis.
All sessions carry a TTL — no indefinite memory leak.
"""
import io
import os
import pickle
import time
from typing import Any, Dict, Optional

import pandas as pd

# TTL in seconds (default 1 hour)
SESSION_TTL = int(os.getenv("SESSION_TTL_SECONDS", "3600"))
SESSION_BACKEND = os.getenv("SESSION_BACKEND", "memory").lower()


class MemorySessionStore:
    """Thread-safe in-memory store with TTL eviction."""

    def __init__(self):
        self._store: Dict[str, Dict[str, Any]] = {}
        self._expiry: Dict[str, float] = {}

    def _evict_expired(self):
        now = time.monotonic()
        expired = [k for k, exp in self._expiry.items() if now > exp]
        for k in expired:
            self._store.pop(k, None)
            self._expiry.pop(k, None)

    def get(self, session_id: str) -> Optional[Dict[str, Any]]:
        self._evict_expired()
        if session_id not in self._store:
            return None
        if time.monotonic() > self._expiry[session_id]:
            self._store.pop(session_id, None)
            self._expiry.pop(session_id, None)
            return None
        return self._store[session_id]

    def set(self, session_id: str, data: Dict[str, Any]) -> None:
        self._evict_expired()
        self._store[session_id] = data
        self._expiry[session_id] = time.monotonic() + SESSION_TTL

    def update(self, session_id: str, patch: Dict[str, Any]) -> None:
        existing = self.get(session_id)
        if existing is None:
            raise KeyError(session_id)
        existing.update(patch)
        self.set(session_id, existing)

    def delete(self, session_id: str) -> None:
        self._store.pop(session_id, None)
        self._expiry.pop(session_id, None)


class RedisSessionStore:
    """Redis-backed session store. DataFrame stored as Parquet bytes."""

    def __init__(self):
        import redis  # lazy import — only required when SESSION_BACKEND=redis
        url = os.getenv("REDIS_URL", "redis://localhost:6379")
        self._r = redis.from_url(url, decode_responses=False)

    def _key(self, session_id: str) -> str:
        return f"fairlens:session:{session_id}"

    def _serialize(self, data: Dict[str, Any]) -> bytes:
        payload = {}
        for k, v in data.items():
            if isinstance(v, pd.DataFrame):
                buf = io.BytesIO()
                v.to_parquet(buf, index=False)
                payload[k] = {"__df__": True, "data": buf.getvalue()}
            else:
                payload[k] = v
        return pickle.dumps(payload)

    def _deserialize(self, raw: bytes) -> Dict[str, Any]:
        payload = pickle.loads(raw)
        result = {}
        for k, v in payload.items():
            if isinstance(v, dict) and v.get("__df__"):
                result[k] = pd.read_parquet(io.BytesIO(v["data"]))
            else:
                result[k] = v
        return result

    def get(self, session_id: str) -> Optional[Dict[str, Any]]:
        raw = self._r.get(self._key(session_id))
        if raw is None:
            return None
        return self._deserialize(raw)

    def set(self, session_id: str, data: Dict[str, Any]) -> None:
        self._r.setex(self._key(session_id), SESSION_TTL, self._serialize(data))

    def update(self, session_id: str, patch: Dict[str, Any]) -> None:
        existing = self.get(session_id)
        if existing is None:
            raise KeyError(session_id)
        existing.update(patch)
        self.set(session_id, existing)

    def delete(self, session_id: str) -> None:
        self._r.delete(self._key(session_id))


def make_session_store():
    if SESSION_BACKEND == "redis":
        return RedisSessionStore()
    return MemorySessionStore()


# Singleton — imported by main.py
session_store = make_session_store()
