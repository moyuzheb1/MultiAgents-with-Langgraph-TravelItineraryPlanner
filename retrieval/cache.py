"""缓存层：Redis 优先，无 Redis 时回退到进程内 TTL 缓存。"""
import json
import os
import time
from typing import Any, Optional


class CacheError(Exception):
    pass


class InMemoryCache:
    """进程内 TTL 缓存（无外部依赖，默认使用）。"""

    def __init__(self):
        self._store: dict[str, tuple[float, Any]] = {}

    def get(self, key: str) -> Optional[Any]:
        hit = self._store.get(key)
        if not hit:
            return None
        expire_at, value = hit
        if expire_at and time.time() > expire_at:
            self._store.pop(key, None)
            return None
        return value

    def set(self, key: str, value: Any, ttl: int = 900) -> None:
        self._store[key] = (time.time() + ttl, value)

    @property
    def backend(self) -> str:
        return "memory"


class RedisCache:
    """Redis 缓存。需要 redis 包与 REDIS_URL 环境变量。"""

    def __init__(self, url: str):
        import redis  # 延迟导入，避免无 Redis 时启动失败

        self._client = redis.Redis.from_url(url, decode_responses=True)
        self._client.ping()

    def get(self, key: str) -> Optional[Any]:
        raw = self._client.get(key)
        return json.loads(raw) if raw else None

    def set(self, key: str, value: Any, ttl: int = 900) -> None:
        self._client.setex(key, ttl, json.dumps(value, ensure_ascii=False))

    @property
    def backend(self) -> str:
        return "redis"


def get_cache(ttl: int = 900):
    """按环境选择缓存后端：配置了 REDIS_URL 且连通则用 Redis，否则用内存缓存。"""
    url = os.getenv("REDIS_URL", "").strip()
    if url:
        try:
            return RedisCache(url)
        except Exception:
            pass
    return InMemoryCache()
