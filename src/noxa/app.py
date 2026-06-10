from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse

from noxa.cache import CacheStore
from noxa.config import get_settings
from noxa.crawler import DeepCrawler
from noxa.fetcher import PageFetcher, fetch_cache_key
from noxa.pipeline import Pipeline
from noxa.proxies import ProxyRotator, redact_proxy
from noxa.schemas import (
    ContentSelectRequest,
    ContentSelectResponse,
    ErrorDetail,
    SearchResultItem,
    TimingMs,
    WebAnswerRequest,
    WebAnswerResponse,
    WebCrawlRequest,
    WebCrawlResponse,
    WebFetchRequest,
    WebFetchResponse,
    WebSearchRequest,
    WebSearchResponse,
)
from noxa.search.ddgs import DDGSSearchProvider, search_cache_key
from noxa.ml_deps import MLDependencyError
from noxa.security import SSRFError

from noxa.logging_config import setup_logging
from noxa.logging_middleware import RequestLoggingMiddleware
from noxa.model_bootstrap import bootstrap_models
from noxa.runtime.registry import RuntimeRegistry
from noxa.openapi_metadata import OPENAPI_DESCRIPTION, OPENAPI_TAGS
from noxa.request_context import log_prefix

logger = logging.getLogger(__name__)

setup_logging()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    logger.info("Noxa booting host=%s port=%d", settings.host, settings.port)
    proxy_rotator = ProxyRotator(settings.proxies)
    if proxy_rotator.enabled:
        logger.info(
            "Proxies enabled: %s",
            [redact_proxy(p) for p in proxy_rotator.all()],
        )
    cache = CacheStore(settings)
    purged = await cache.purge_fetch_cache()
    if purged:
        logger.info("Purged %d stale fetch cache entries", purged)
    registry = RuntimeRegistry.from_settings(settings)
    fetcher = PageFetcher(settings, proxy_rotator)
    await fetcher.start()
    try:
        await bootstrap_models(settings, registry)
        search = DDGSSearchProvider(proxy_rotator)
        pipeline = Pipeline(settings, cache, search, fetcher, registry)
        crawler = DeepCrawler(fetcher, settings)

        app.state.settings = settings
        app.state.cache = cache
        app.state.registry = registry
        app.state.fetcher = fetcher
        app.state.search = search
        app.state.pipeline = pipeline
        app.state.crawler = crawler

        logger.info("Noxa started")
        yield
    finally:
        await fetcher.stop()
        cache.close()
        logger.info("Noxa stopped")


app = FastAPI(
    title="Noxa",
    description=OPENAPI_DESCRIPTION,
    version="0.1.0",
    lifespan=lifespan,
    openapi_tags=OPENAPI_TAGS,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)
app.add_middleware(RequestLoggingMiddleware)


@app.get("/", include_in_schema=False)
async def root() -> RedirectResponse:
    return RedirectResponse(url="/docs")


@app.get("/health", tags=["meta"], summary="Health check")
async def health() -> dict[str, str]:
    return {"status": "ok", "service": "noxa", "version": "0.1.0"}


@app.post(
    "/web_search",
    response_model=WebSearchResponse,
    tags=["search"],
    summary="Search the web",
    description="Run a web search and return ranked results. Results are cached by query.",
)
async def web_search(req: WebSearchRequest) -> WebSearchResponse:
    cache = app.state.cache
    search: DDGSSearchProvider = app.state.search
    settings = app.state.settings
    logger.info(
        "%s/web_search query=%r max_results=%d region=%s",
        log_prefix(),
        req.query,
        req.max_results,
        req.region,
    )
    t0 = time.perf_counter()
    key = search_cache_key(req.provider, req.query, req.region, req.max_results)
    cached = await cache.get(key)
    if cached:
        results = [SearchResultItem(**r) for r in cached]
        logger.info("%s/web_search cache_hit results=%d", log_prefix(), len(results))
    else:
        try:
            results = await search.search(
                req.query, req.max_results, req.region, req.safe_search
            )
        except Exception as exc:
            logger.exception("%s/web_search failed", log_prefix())
            raise HTTPException(
                status_code=502,
                detail=ErrorDetail(
                    code="SEARCH_PROVIDER_FAILED",
                    message=str(exc),
                    retryable=True,
                ).model_dump(),
            ) from exc
        await cache.set(
            key,
            [r.model_dump() for r in results],
            settings.search_ttl_seconds,
        )
    elapsed = int((time.perf_counter() - t0) * 1000)
    return WebSearchResponse(
        query=req.query,
        provider=req.provider,
        results=results,
        timing_ms=TimingMs(search=elapsed, total=elapsed),
    )


@app.post(
    "/web_fetch",
    response_model=WebFetchResponse,
    tags=["fetch"],
    summary="Fetch a single URL",
    description="Fetch a public URL and return extracted markdown text.",
)
async def web_fetch(req: WebFetchRequest) -> WebFetchResponse:
    cache = app.state.cache
    fetcher: PageFetcher = app.state.fetcher
    settings = app.state.settings
    logger.info("%s/web_fetch url=%s", log_prefix(), req.url)
    try:
        key = fetch_cache_key(req.url)
        cached = await cache.get(key)
        if cached:
            logger.info("%s/web_fetch cache_hit url=%s", log_prefix(), req.url)
            data = dict(cached)
            data.pop("timing_ms", None)
            page = WebFetchResponse(
                **data,
                timing_ms=TimingMs(fetch=0, total=0),
            )
            page.text = page.text[: req.max_chars]
            return page
        resp = await fetcher.fetch_response(req.url, req.max_chars)
        await cache.set(key, resp.model_dump(), settings.fetch_ttl_seconds)
        return resp
    except SSRFError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.post(
    "/web_crawl",
    response_model=WebCrawlResponse,
    tags=["crawl"],
    summary="Deep crawl from seed URLs",
    description="Crawl pages starting from one or more seed URLs, optionally scored by a query.",
)
async def web_crawl(req: WebCrawlRequest) -> WebCrawlResponse:
    crawler: DeepCrawler = app.state.crawler
    logger.info(
        "%s/web_crawl seeds=%d max_pages=%d query=%r",
        log_prefix(),
        len(req.urls),
        req.max_pages,
        req.query,
    )
    try:
        return await crawler.crawl(
            req.urls,
            query=req.query,
            max_pages=req.max_pages,
            max_depth=req.max_depth,
            include_external=req.include_external,
        )
    except SSRFError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.post(
    "/content_select",
    response_model=ContentSelectResponse,
    tags=["retrieval"],
    summary="Select relevant passages",
    description="Chunk documents and run hybrid retrieval + reranking to pick the best passages.",
)
async def content_select(req: ContentSelectRequest) -> ContentSelectResponse:
    pipeline: Pipeline = app.state.pipeline
    logger.info(
        "%s/content_select query=%r docs=%d mode=%s",
        log_prefix(),
        req.query,
        len(req.documents),
        req.mode,
    )
    try:
        return await pipeline.select_from_documents(
            req.query,
            req.documents,
            mode=req.mode,
            token_budget=req.token_budget,
            return_debug=req.return_debug,
        )
    except MLDependencyError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post(
    "/web_answer",
    response_model=WebAnswerResponse,
    tags=["answer"],
    summary="Search, fetch, and answer",
    description=(
        "Full pipeline: search the web, fetch top pages, select passages, "
        "and generate a grounded answer with citations. "
        "Set `return_debug: true` to write a debug dump under `dump/` "
        "(see `debug.dump_dir` and `debug.meta_md` in the response)."
    ),
)
async def web_answer(req: WebAnswerRequest) -> WebAnswerResponse:
    pipeline: Pipeline = app.state.pipeline
    logger.info(
        "%s/web_answer query=%r mode=%s max_pages=%s",
        log_prefix(),
        req.query,
        req.mode,
        req.max_pages,
    )
    try:
        return await pipeline.web_answer(
            req.query,
            mode=req.mode,
            max_search_results=req.max_search_results,
            max_pages=req.max_pages,
            return_sources=req.return_sources,
            return_debug=req.return_debug,
        )
    except MLDependencyError as exc:
        logger.error("%s/web_answer ml_unavailable: %s", log_prefix(), exc)
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        msg = str(exc)
        logger.exception("%s/web_answer failed: %s", log_prefix(), msg)
        if "no pages could be fetched" in msg.lower():
            raise HTTPException(
                status_code=502,
                detail=ErrorDetail(
                    code="NO_FETCHABLE_SOURCES",
                    message=msg,
                    retryable=True,
                ).model_dump(),
            ) from exc
        raise HTTPException(
            status_code=502,
            detail=ErrorDetail(
                code="PIPELINE_FAILED",
                message=msg,
                retryable=True,
            ).model_dump(),
        ) from exc


def run() -> None:
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "noxa.app:app",
        host=settings.host,
        port=settings.port,
        reload=False,
    )
