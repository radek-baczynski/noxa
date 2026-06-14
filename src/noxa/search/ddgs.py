from __future__ import annotations

import asyncio
import hashlib
import logging

from ddgs import DDGS
from ddgs.engines import ENGINES

from noxa.proxies import ProxyRotator
from noxa.request_context import log_prefix
from noxa.schemas import SearchResultItem
from noxa.search.base import SearchProvider

logger = logging.getLogger(__name__)


def _resolve_text_engines(backend: str | None) -> list[str]:
    """Map a ddgs backend setting to engine names (public ENGINES registry)."""
    available = sorted(ENGINES["text"].keys())
    if not backend or backend in {"auto", "all"}:
        return available
    requested = [part.strip() for part in backend.split(",") if part.strip()]
    return [name for name in requested if name in ENGINES["text"]]


class DDGSSearchProvider(SearchProvider):
    def __init__(self, proxy_rotator: ProxyRotator) -> None:
        self._proxy_rotator = proxy_rotator

    async def search(
        self,
        query: str,
        max_results: int,
        region: str | None = None,
        safe_search: str | None = None,
        backend: str | None = None,
    ) -> list[SearchResultItem]:
        proxy = self._proxy_rotator.next()
        backend_eff = backend or "auto"
        region_eff = region or "wt-wt"
        engines = _resolve_text_engines(backend)
        logger.info(
            "%sddgs search query=%r max_results=%d backend=%s engines=%s region=%s proxy=%s",
            log_prefix(),
            query,
            max_results,
            backend_eff,
            ",".join(engines) if engines else "none",
            region_eff,
            "yes" if proxy else "no",
        )
        return await asyncio.to_thread(
            self._search_sync,
            query,
            max_results,
            region_eff,
            safe_search,
            proxy,
            backend_eff,
        )

    def _search_sync(
        self,
        query: str,
        max_results: int,
        region: str,
        safe_search: str | None,
        proxy: str | None,
        backend: str,
    ) -> list[SearchResultItem]:
        kwargs: dict = {
            "max_results": max_results,
            "region": region,
            "safesearch": safe_search or "moderate",
            "backend": backend,
        }
        raw = DDGS(proxy=proxy).text(query, **kwargs)
        results: list[SearchResultItem] = []
        for i, item in enumerate(raw, start=1):
            results.append(
                SearchResultItem(
                    rank=i,
                    title=item.get("title"),
                    url=item.get("href") or item.get("url", ""),
                    snippet=item.get("body") or item.get("snippet"),
                    source_provider="ddgs",
                )
            )
        logger.info(
            "%sddgs search done backend=%s merged_results=%d",
            log_prefix(),
            backend,
            len(results),
        )
        return results


def search_cache_key(
    provider: str,
    query: str,
    region: str | None,
    max_results: int,
    backend: str | None = None,
) -> str:
    raw = f"{provider}|{query}|{region or 'wt-wt'}|{max_results}|{backend or ''}"
    return hashlib.sha256(raw.encode()).hexdigest()
