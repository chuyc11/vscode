"""Redis + zstd compression storage layer for pipeline state persistence.

Engineering red lines
---------------------
1. All large payloads (facts, graph, draft, report) are stored in Redis,
   never in LangGraph state — only Redis key pointers are in state.
2. Zstandard (zstd) level 3 compression: minimal CPU overhead, significant
   memory savings for JSON-serialized Pydantic models.
3. Absolute TTL (3600s) on ALL keys — no orphaned data, no memory leaks.
4. Lifecycle finally hooks call cleanup() to delete all temporary keys.
5. Redis must be configured with maxmemory-policy volatile-lru so that
   TTL-expired keys are evicted first under memory pressure.
"""

import uuid
import logging

import redis
import zstandard
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# REDIS CONFIG CONTRACT
# redis.conf must set:
#   maxmemory-policy volatile-lru
# This ensures TTL-expired keys are evicted first under memory pressure.
# Our code sets absolute TTL (3600s) on ALL keys — no orphaned data.


class StateStore:
    """Redis-backed state store with zstd compression and absolute TTL."""

    TTL_SECONDS = 3600  # 1 hour absolute TTL

    def __init__(self, redis_url: str = "redis://localhost:6379/0"):
        self._redis = redis.Redis.from_url(redis_url, decode_responses=False)
        self._zstd = zstandard.ZstdCompressor(level=3)  # level 3 = minimal CPU
        self._decompressor = zstandard.ZstdDecompressor()

    def put(self, obj: BaseModel) -> str:
        """Serialize → zstd compress → store in Redis, return key."""
        key = f"pipeline:{uuid.uuid4().hex}"
        raw = obj.model_dump_json().encode("utf-8")
        compressed = self._zstd.compress(raw)
        self._redis.setex(key, self.TTL_SECONDS, compressed)
        logger.debug("state_store.put: key=%s, raw_size=%d, compressed=%d",
                      key, len(raw), len(compressed))
        return key

    def get(self, key: str, model_class: type) -> BaseModel:
        """Retrieve from Redis → zstd decompress → deserialize."""
        compressed = self._redis.get(key)
        if compressed is None:
            raise KeyError(f"Key expired or missing: {key}")
        raw = self._decompressor.decompress(compressed)
        return model_class.model_validate_json(raw)

    def cleanup(self, *keys: str) -> None:
        """Lifecycle finally hook: delete all temporary keys."""
        if keys:
            deleted = self._redis.delete(*keys)
            logger.info("state_store.cleanup: deleted %d/%d keys", deleted, len(keys))
