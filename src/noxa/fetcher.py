from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from typing import Any
from urllib.parse import urlparse, urlunparse

from crawl4ai import AsyncWebCrawler, BrowserConfig, CacheMode, CrawlerRunConfig

from noxa.config import Settings
from noxa.proxies import ProxyRotator
from noxa.request_context import log_prefix
from noxa.schemas import ExtractedPage, PageMetadata, TimingMs, WebFetchResponse
from noxa.security import validate_public_url
from noxa.text_clean import clean_page_markdown

logger = logging.getLogger(__name__)


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _truncate_page_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars]


def _normalize_url(url: str) -> str:
    parsed = urlparse(url.strip())
    netloc = parsed.netloc.lower()
    path = parsed.path.rstrip("/") or "/"
    return urlunparse((parsed.scheme.lower(), netloc, path, parsed.params, parsed.query, ""))


def _result_url_keys(result: Any) -> list[str]:
    keys: list[str] = []
    for attr in ("url", "redirected_url", "final_url"):
        value = getattr(result, attr, None)
        if value:
            keys.append(value)
    if result.metadata:
        for meta_key in ("url", "source_url", "seed_url", "initial_url"):
            value = result.metadata.get(meta_key)
            if value:
                keys.append(value)
    return keys


def _match_result_to_url(requested_url: str, results: list[Any], used: set[int]) -> Any | None:
    requested_norm = _normalize_url(requested_url)
    for idx, result in enumerate(results):
        if idx in used or not result.success:
            continue
        for key in _result_url_keys(result):
            if _normalize_url(key) == requested_norm:
                used.add(idx)
                return result
    return None


def _extract_markdown(result: Any) -> str:
    if result.markdown is None:
        return ""
    fit = getattr(result.markdown, "fit_markdown", None) or ""
    raw = getattr(result.markdown, "raw_markdown", None) or ""
    return fit or raw


def _page_from_crawl_result(
    url: str, result: Any, max_chars: int
) -> ExtractedPage:
    markdown = clean_page_markdown(_extract_markdown(result))
    title = result.metadata.get("title") if result.metadata else None
    text = _truncate_page_text(markdown, max_chars)
    final_url = result.url or url
    cleaned_html = result.cleaned_html
    if not cleaned_html and result.markdown is not None:
        cleaned_html = getattr(result.markdown, "fit_html", None)
    if not cleaned_html:
        cleaned_html = result.fit_html
    return ExtractedPage(
        source_id=hashlib.sha256(final_url.encode()).hexdigest()[:16],
        url=url,
        final_url=final_url,
        title=title,
        text=text,
        content_hash=content_hash(text),
        extractor="crawl4ai",
        html=result.html or None,
        cleaned_html=cleaned_html or None,
    )


def page_cache_payload(page: ExtractedPage, max_chars: int) -> dict:
    """Serialize page for SQLite cache, truncating large text/html fields."""
    data = page.model_dump()
    for field in ("text", "html", "cleaned_html"):
        value = data.get(field)
        if value and len(value) > max_chars:
            data[field] = value[:max_chars]
    return data


def fetch_cache_key(url: str) -> str:
    return hashlib.sha256(url.encode()).hexdigest()


def page_from_cache(cached: dict) -> ExtractedPage:
    """Build ExtractedPage from cached WebFetchResponse or ExtractedPage dict."""
    if "source_id" in cached:
        return ExtractedPage(**cached)
    final_url = cached.get("final_url") or cached["url"]
    return ExtractedPage(
        source_id=hashlib.sha256(final_url.encode()).hexdigest()[:16],
        url=cached["url"],
        final_url=final_url,
        title=cached.get("title"),
        text=cached["text"],
        content_hash=cached.get("content_hash", content_hash(cached["text"])),
        extractor=cached.get("extractor", "crawl4ai"),
        html=cached.get("html"),
        cleaned_html=cached.get("cleaned_html"),
    )


class PageFetcher:
    def __init__(self, settings: Settings, proxy_rotator: ProxyRotator) -> None:
        self.settings = settings
        self.proxy_rotator = proxy_rotator
        self._crawler: AsyncWebCrawler | None = None

    def _browser_config(self) -> BrowserConfig:
        proxy = self.proxy_rotator.next()
        kwargs: dict[str, Any] = {"headless": True, "user_agent": self.settings.user_agent}
        if proxy:
            kwargs["proxy"] = proxy
        return BrowserConfig(**kwargs)

    async def start(self) -> None:
        logger.info("PageFetcher starting (proxy=%s)", self.proxy_rotator.enabled)
        self._crawler = AsyncWebCrawler(config=self._browser_config())
        await self._crawler.__aenter__()
        logger.info("PageFetcher ready")

    async def stop(self) -> None:
        if self._crawler is not None:
            logger.info("PageFetcher stopping")
            await self._crawler.__aexit__(None, None, None)
            self._crawler = None

    def _run_config(self, timeout_ms: int) -> CrawlerRunConfig:
        return CrawlerRunConfig(
            cache_mode=CacheMode.BYPASS,
            page_timeout=timeout_ms,
            stream=False,
        )

    async def fetch_url(self, url: str, timeout_ms: int | None = None) -> ExtractedPage:
        validate_public_url(url)
        if self._crawler is None:
            raise RuntimeError("PageFetcher not started")
        timeout = timeout_ms or self.settings.per_page_timeout_ms
        t0 = time.perf_counter()
        logger.debug("%sfetch url=%s timeout=%dms", log_prefix(), url, timeout)
        result = await self._crawler.arun(url, config=self._run_config(timeout))
        elapsed = int((time.perf_counter() - t0) * 1000)
        if not result.success:
            logger.warning(
                "%sfetch failed url=%s error=%s latency=%dms",
                log_prefix(),
                url,
                result.error_message,
                elapsed,
            )
            raise RuntimeError(result.error_message or f"Failed to fetch {url}")
        page = _page_from_crawl_result(url, result, self.settings.max_chars_per_page)
        logger.info(
            "%sfetch ok url=%s chars=%d html=%d cleaned=%d latency=%dms",
            log_prefix(),
            url,
            len(page.text),
            len(page.html or ""),
            len(page.cleaned_html or ""),
            elapsed,
        )
        return page

    async def fetch_many(
        self, urls: list[str], timeout_ms: int | None = None
    ) -> tuple[list[ExtractedPage], list[str]]:
        from noxa.security import validate_public_urls as _validate

        safe_urls = _validate(urls)
        if self._crawler is None:
            raise RuntimeError("PageFetcher not started")
        timeout = timeout_ms or self.settings.per_page_timeout_ms
        logger.info(
            "%sfetch_many start urls=%d timeout=%dms",
            log_prefix(),
            len(safe_urls),
            timeout,
        )
        t0 = time.perf_counter()
        results = await self._crawler.arun_many(
            safe_urls, config=self._run_config(timeout)
        )
        pages: list[ExtractedPage] = []
        failed: list[str] = []
        used: set[int] = set()
        unmatched: list[str] = []
        for url in safe_urls:
            result = _match_result_to_url(url, results, used)
            if result is None:
                unmatched.append(url)
                continue
            pages.append(
                _page_from_crawl_result(url, result, self.settings.max_chars_per_page)
            )

        if unmatched:
            logger.info(
                "%sfetch_many refetching %d unmatched urls",
                log_prefix(),
                len(unmatched),
            )

            async def _refetch(requested: str) -> tuple[str, ExtractedPage | None]:
                try:
                    return requested, await self.fetch_url(requested, timeout_ms=timeout)
                except RuntimeError:
                    return requested, None

            refetched = await asyncio.gather(*[_refetch(u) for u in unmatched])
            for url, page in refetched:
                if page is None:
                    logger.warning(
                        "%sfetch_many failed url=%s",
                        log_prefix(),
                        url,
                    )
                    failed.append(url)
                else:
                    pages.append(page)
        elapsed = int((time.perf_counter() - t0) * 1000)
        logger.info(
            "%sfetch_many done ok=%d failed=%d latency=%dms",
            log_prefix(),
            len(pages),
            len(failed),
            elapsed,
        )
        return pages, failed

    async def fetch_response(self, url: str, max_chars: int = 80_000) -> WebFetchResponse:
        t0 = time.perf_counter()
        page = await self.fetch_url(url)
        text = page.text[:max_chars]
        elapsed = int((time.perf_counter() - t0) * 1000)
        return WebFetchResponse(
            url=page.url,
            final_url=page.final_url,
            title=page.title,
            text=text,
            metadata=PageMetadata(),
            extractor=page.extractor,
            content_hash=page.content_hash,
            timing_ms=TimingMs(fetch=elapsed, extract=0, total=elapsed),
        )
