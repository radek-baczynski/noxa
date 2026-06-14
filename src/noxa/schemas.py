from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from noxa.config import AnswerSize, RuntimeMode, SafeSearch, SearchProvider


class TimingMs(BaseModel):
    search: int | None = None
    fetch: int | None = None
    extract: int | None = None
    crawl: int | None = None
    select: int | None = None
    answer: int | None = None
    total: int | None = None


class AnswerLlmPerf(BaseModel):
    """LLM answer throughput from backend perf counters (llama.cpp prefill/decode)."""

    prefill_ms: float = 0.0
    prefill_tokens: int = 0
    prefill_tps: float = 0.0
    decode_ms: float = 0.0
    decode_tokens: int = 0
    decode_tps: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0


class ModelInferMs(BaseModel):
    """Wall-clock time and item counts for model forward passes only."""

    embedding_time: int = 0
    embedding_calls: int = 0
    rerank_time: int = 0
    rerank_calls: int = 0
    answer_time: int = 0
    answer_calls: int = 0
    answer_llm: AnswerLlmPerf | None = None


class ErrorDetail(BaseModel):
    code: str
    message: str
    retryable: bool = False


# --- Search ---


class WebSearchRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "query": "python asyncio tutorial",
                    "max_results": 5,
                    "provider": "ddgs",
                    "region": "wt-wt",
                    "safe_search": "moderate",
                }
            ]
        }
    )

    query: str
    max_results: int = 8
    provider: SearchProvider = SearchProvider.DDGS
    region: str = "wt-wt"
    safe_search: SafeSearch = SafeSearch.MODERATE


class SearchResultItem(BaseModel):
    rank: int
    title: str | None
    url: str
    snippet: str | None
    source_provider: str = "ddgs"


class WebSearchResponse(BaseModel):
    query: str
    provider: str
    results: list[SearchResultItem]
    timing_ms: TimingMs = Field(default_factory=TimingMs)


# --- Fetch ---


class WebFetchRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [{"url": "https://www.python.org", "max_chars": 20000}]
        }
    )

    url: str
    max_chars: int = 80_000


class PageMetadata(BaseModel):
    author: str | None = None
    published_at: str | None = None
    language: str | None = None


class WebFetchResponse(BaseModel):
    url: str
    final_url: str
    title: str | None
    text: str
    metadata: PageMetadata = Field(default_factory=PageMetadata)
    extractor: str = "crawl4ai"
    content_hash: str
    timing_ms: TimingMs = Field(default_factory=TimingMs)


# --- Crawl ---


class WebCrawlRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "urls": ["https://docs.python.org/3/"],
                    "query": "asyncio",
                    "max_pages": 5,
                    "max_depth": 1,
                    "include_external": False,
                }
            ]
        }
    )

    urls: list[str]
    query: str | None = None
    max_pages: int = 10
    max_depth: int = 2
    include_external: bool = False


class CrawledPage(BaseModel):
    url: str
    title: str | None
    text: str
    depth: int
    score: float
    seed_url: str


class WebCrawlResponse(BaseModel):
    seed_urls: list[str]
    query: str | None
    pages: list[CrawledPage]
    failed_urls: list[str] = Field(default_factory=list)
    timing_ms: TimingMs = Field(default_factory=TimingMs)


# --- Content select ---


class DocumentInput(BaseModel):
    id: str | None = None
    url: str | None = None
    title: str | None = None
    text: str


class ContentSelectRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "query": "asyncio event loop",
                    "mode": "default",
                    "token_budget": 3500,
                    "return_debug": True,
                    "documents": [
                        {
                            "id": "doc1",
                            "title": "Asyncio docs",
                            "text": "The asyncio event loop runs coroutines and handles I/O.",
                        },
                        {
                            "id": "doc2",
                            "title": "Unrelated",
                            "text": "Bananas are yellow fruits.",
                        },
                    ],
                }
            ]
        }
    )

    query: str
    documents: list[DocumentInput]
    mode: RuntimeMode = RuntimeMode.DEFAULT
    token_budget: int = 3500
    return_debug: bool = False


class SelectedPassage(BaseModel):
    source_id: str
    url: str | None
    title: str | None
    passage_id: str
    text: str
    score: float
    selection_reason: str = "high reranker score"


class SelectionDebug(BaseModel):
    total_passages: int
    bm25_candidates: int
    embedding_candidates: int
    merged_candidates: int
    reranked_candidates: int
    final_passages: int


class ContentSelectResponse(BaseModel):
    query: str
    selected_passages: list[SelectedPassage]
    debug: dict[str, Any] | None = None


# --- Web answer ---


class WebAnswerRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "query": "What is Python asyncio?",
                    "mode": "default",
                    "answer_size": "medium",
                    "return_sources": True,
                    "return_debug": True,
                },
                {
                    "query": "What is Python asyncio?",
                    "mode": "fast",
                    "max_search_results": 8,
                    "max_pages": 4,
                    "return_sources": True,
                    "return_debug": True,
                },
            ]
        }
    )

    query: str
    mode: RuntimeMode = RuntimeMode.DEFAULT
    answer_size: AnswerSize = AnswerSize.MEDIUM
    search_provider: SearchProvider = SearchProvider.DDGS
    max_search_results: int | None = None
    max_pages: int | None = None
    max_depth: int | None = None
    return_sources: bool = True
    return_debug: bool = False


class Citation(BaseModel):
    source_id: int
    url: str | None
    title: str | None
    supports: str | None = None


class AnswerPassage(BaseModel):
    passage_id: str
    text: str
    score: float


class AnswerSource(BaseModel):
    id: int
    url: str | None
    title: str | None
    selected_passages: list[AnswerPassage] = Field(default_factory=list)


class WebAnswerResponse(BaseModel):
    query: str
    answer: str
    confidence: str = "medium"
    abstained: bool = False
    citations: list[Citation] = Field(default_factory=list)
    sources: list[AnswerSource] = Field(default_factory=list)
    timing_ms: TimingMs = Field(default_factory=TimingMs)
    model_infer: ModelInferMs = Field(default_factory=ModelInferMs)
    debug: dict[str, Any] | None = None


# --- Internal domain models ---


class ExtractedPage(BaseModel):
    source_id: str
    url: str
    final_url: str
    title: str | None
    text: str
    content_hash: str
    extractor: str = "crawl4ai"
    source_rank: int = 0
    html: str | None = None
    cleaned_html: str | None = None


class Passage(BaseModel):
    passage_id: str
    source_id: str
    url: str | None
    title: str | None
    text: str
    start_char: int
    end_char: int
    token_count: int
    source_rank: int = 0


class ScoredPassage(BaseModel):
    passage: Passage
    bm25_score: float | None = None
    embedding_score: float | None = None
    merged_score: float | None = None
    rerank_score: float | None = None


def utc_now_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
