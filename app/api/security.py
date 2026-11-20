"""Optional API-key auth + simple per-IP sliding-window rate limit (in-memory, single process)."""

import hmac
import threading
import time
from collections import defaultdict, deque

from fastapi import Depends, Header, HTTPException, Request, status

from app.config import Settings, get_settings


def require_api_key(x_api_key: str | None = Header(default=None),
                    settings: Settings = Depends(get_settings)) -> None:
    if settings.api_key is None:
        return
    if not x_api_key or not hmac.compare_digest(x_api_key, settings.api_key.get_secret_value()):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid or missing X-API-Key")


class RateLimiter:
    def __init__(self, per_minute: int):
        self.per_minute = per_minute
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def check(self, key: str) -> None:
        if self.per_minute <= 0:
            return
        now = time.monotonic()
        with self._lock:
            q = self._hits[key]
            while q and now - q[0] > 60:
                q.popleft()
            if len(q) >= self.per_minute:
                retry = int(60 - (now - q[0])) + 1
                raise HTTPException(status.HTTP_429_TOO_MANY_REQUESTS, "Rate limit exceeded",
                                    headers={"Retry-After": str(retry)})
            q.append(now)


def rate_limited(request: Request) -> None:
    limiter: RateLimiter = request.app.state.rate_limiter
    client = request.client.host if request.client else "unknown"
    limiter.check(f"{client}:{request.url.path}")
