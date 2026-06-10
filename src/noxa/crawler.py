from __future__ import annotations

import logging
import re
import time

from crawl4ai import CacheMode, CrawlerRunConfig
from crawl4ai.content_scraping_strategy import LXMLWebScrapingStrategy
from crawl4ai.deep_crawling import BestFirstCrawlingStrategy
from crawl4ai.deep_crawling.filters import ContentTypeFilter, DomainFilter, FilterChain
from crawl4ai.deep_crawling.scorers import KeywordRelevanceScorer

from noxa.config import Settings
from noxa.fetcher import PageFetcher, _truncate_page_text
from noxa.request_context import log_prefix
from noxa.schemas import CrawledPage, TimingMs, WebCrawlResponse
from noxa.security import validate_public_urls

logger = logging.getLogger(__name__)


def _keywords_from_query(query: str | None) -> list[str]:
    if not query:
        return []
    return [w for w in re.split(r"\W+", query.lower()) if len(w) > 2]


class DeepCrawler:
    def __init__(self, fetcher: PageFetcher, settings: Settings) -> None:
        self.fetcher = fetcher
        self.settings = settings

    async def crawl(
        self,
        urls: list[str],
        query: str | None = None,
        max_pages: int | None = None,
        max_depth: int | None = None,
        include_external: bool | None = None,
    ) -> WebCrawlResponse:
        seed_urls = validate_public_urls(urls)
        max_pages = max_pages if max_pages is not None else self.settings.crawl_max_pages
        max_depth = max_depth if max_depth is not None else self.settings.crawl_max_depth
        include_external = (
            include_external
            if include_external is not None
            else self.settings.crawl_include_external
        )
        logger.info(
            "%scrawl start seeds=%d max_pages=%d max_depth=%d query=%r",
            log_prefix(),
            len(seed_urls),
            max_pages,
            max_depth,
            query,
        )

        if self.fetcher._crawler is None:
            raise RuntimeError("PageFetcher not started")

        allowed_domains: list[str] = []
        for url in seed_urls:
            from urllib.parse import urlparse

            host = urlparse(url).hostname
            if host:
                allowed_domains.append(host)

        filters = [ContentTypeFilter(allowed_types=["text/html", "application/xhtml+xml"])]
        if allowed_domains and not include_external:
            filters.insert(0, DomainFilter(allowed_domains=allowed_domains))

        filter_chain = FilterChain(filters)
        keywords = _keywords_from_query(query)
        url_scorer = KeywordRelevanceScorer(keywords=keywords or ["page"], weight=0.7)

        config = CrawlerRunConfig(
            deep_crawl_strategy=BestFirstCrawlingStrategy(
                max_depth=max_depth,
                include_external=include_external,
                filter_chain=filter_chain,
                url_scorer=url_scorer,
                max_pages=max_pages,
            ),
            scraping_strategy=LXMLWebScrapingStrategy(),
            cache_mode=CacheMode.BYPASS,
            stream=True,
            page_timeout=self.settings.per_page_timeout_ms,
        )

        t0 = time.perf_counter()
        pages: list[CrawledPage] = []
        failed: list[str] = []
        seen: set[str] = set()

        for seed in seed_urls:
            async for result in await self.fetcher._crawler.arun(seed, config=config):
                if not result.success:
                    if result.url and result.url not in seen:
                        failed.append(result.url)
                    continue
                final_url = result.url or seed
                if final_url in seen:
                    continue
                seen.add(final_url)
                markdown = ""
                if result.markdown is not None:
                    markdown = (
                        getattr(result.markdown, "fit_markdown", None)
                        or getattr(result.markdown, "raw_markdown", None)
                        or ""
                    )
                title = result.metadata.get("title") if result.metadata else None
                depth = int(result.metadata.get("depth", 0)) if result.metadata else 0
                score = float(result.metadata.get("score", 0.0)) if result.metadata else 0.0
                pages.append(
                    CrawledPage(
                        url=final_url,
                        title=title,
                        text=_truncate_page_text(
                            markdown.strip(), self.settings.max_chars_per_page
                        ),
                        depth=depth,
                        score=score,
                        seed_url=seed,
                    )
                )
                if len(pages) >= max_pages:
                    break
            if len(pages) >= max_pages:
                break

        pages.sort(key=lambda p: p.score, reverse=True)
        elapsed = int((time.perf_counter() - t0) * 1000)
        logger.info(
            "%scrawl done pages=%d failed=%d latency=%dms",
            log_prefix(),
            len(pages),
            len(failed),
            elapsed,
        )
        return WebCrawlResponse(
            seed_urls=seed_urls,
            query=query,
            pages=pages[:max_pages],
            failed_urls=failed,
            timing_ms=TimingMs(crawl=elapsed, total=elapsed),
        )
