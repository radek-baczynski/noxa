from __future__ import annotations

import asyncio
import hashlib
import logging

from ddgs import DDGS

from noxa.proxies import ProxyRotator
from noxa.request_context import log_prefix
from noxa.schemas import SearchResultItem
from noxa.search.base import SearchProvider

logger = logging.getLogger(__name__)


class DDGSSearchProvider(SearchProvider):
    def __init__(self, proxy_rotator: ProxyRotator) -> None:
        self._proxy_rotator = proxy_rotator

    async def search(
        self,
        query: str,
        max_results: int,
        region: str | None = None,
        safe_search: str | None = None,
    ) -> list[SearchResultItem]:
        proxy = self._proxy_rotator.next()
        logger.info(
            "%sddgs search query=%r max_results=%d proxy=%s",
            log_prefix(),
            query,
            max_results,
            "yes" if proxy else "no",
        )
        return await asyncio.to_thread(
            self._search_sync, query, max_results, region, safe_search, proxy
        )

    def _search_sync(
        self,
        query: str,
        max_results: int,
        region: str | None,
        safe_search: str | None,
        proxy: str | None,
    ) -> list[SearchResultItem]:
        kwargs: dict = {"max_results": max_results}
        if region:
            kwargs["region"] = region
        if safe_search:
            kwargs["safe_search"] = safe_search
        if proxy:
            kwargs["proxy"] = proxy
        raw = DDGS().text(query, **kwargs)
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
        logger.info("%sddgs search done results=%d", log_prefix(), len(results))
        return results


def search_cache_key(provider: str, query: str, region: str, max_results: int) -> str:
    raw = f"{provider}|{query}|{region}|{max_results}"
    return hashlib.sha256(raw.encode()).hexdigest()
