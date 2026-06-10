from __future__ import annotations

import itertools
import threading
from urllib.parse import urlparse


def redact_proxy(proxy: str) -> str:
    parsed = urlparse(proxy)
    if parsed.username or parsed.password:
        host = parsed.hostname or ""
        port = f":{parsed.port}" if parsed.port else ""
        return f"{parsed.scheme}://***:***@{host}{port}"
    return proxy


class ProxyRotator:
    def __init__(self, proxies: list[str]) -> None:
        self._proxies = [p.strip() for p in proxies if p.strip()]
        self._cycle = itertools.cycle(self._proxies) if self._proxies else None
        self._lock = threading.Lock()

    @property
    def enabled(self) -> bool:
        return bool(self._proxies)

    def next(self) -> str | None:
        if not self._cycle:
            return None
        with self._lock:
            return next(self._cycle)

    def all(self) -> list[str]:
        return list(self._proxies)
