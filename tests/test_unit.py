from __future__ import annotations

import pytest

from noxa.chunker import chunk_pages, count_tokens
from noxa.model_stats import reset_model_infer_stats
from noxa.model_timing import timed_model_op
from noxa.config import get_settings
from noxa.fetcher import page_from_cache
from noxa.proxies import ProxyRotator, redact_proxy
from noxa.retrieval.bm25 import bm25_search
from noxa.retrieval.merge import merge_candidates
from noxa.schemas import ExtractedPage, Passage, ScoredPassage
from noxa.security import SSRFError, validate_public_url
from noxa.search.ddgs import _resolve_text_engines, search_cache_key


def test_llm_perf_tps_calculation() -> None:
    from noxa.llm_perf import LlmPerfStats, _tps

    assert _tps(100, 1000.0) == 100.0
    assert _tps(0, 1000.0) == 0.0
    perf = LlmPerfStats(
        prefill_ms=0.0,
        prefill_tokens=50,
        decode_ms=400.0,
        decode_tokens=20,
        decode_tps=_tps(20, 400.0),
        prompt_tokens=50,
        completion_tokens=20,
    )
    assert perf.decode_tps == 50.0
    assert perf.prompt_tokens == 50
    assert perf.completion_tokens == 20


def test_model_infer_stats_answer_llm_in_to_dict() -> None:
    from noxa.llm_perf import LlmPerfStats
    from noxa.model_stats import ModelInferStats

    stats = ModelInferStats()
    stats.set_answer_llm(
        LlmPerfStats(
            prefill_ms=120.5,
            prefill_tokens=300,
            prefill_tps=2489.6,
            decode_ms=80.0,
            decode_tokens=16,
            decode_tps=200.0,
        )
    )
    d = stats.to_dict()
    assert d["answer_llm"]["prefill_tps"] == 2489.6
    assert d["answer_llm"]["decode_tps"] == 200.0


def test_model_infer_stats_accumulate_per_request() -> None:
    reset_model_infer_stats()
    with timed_model_op("embed-model", "infer encode", infer_kind="embedding", infer_calls=3):
        pass
    with timed_model_op("rerank-model", "infer rerank", infer_kind="rerank", infer_calls=4):
        pass
    with timed_model_op("answer-model", "infer generate", infer_kind="answer", infer_calls=1):
        pass
    from noxa.model_stats import get_model_infer_stats

    got = get_model_infer_stats().to_dict()
    assert got["embedding_calls"] == 3
    assert got["rerank_calls"] == 4
    assert got["answer_calls"] == 1
    assert got["embedding_time"] >= 0
    assert got["rerank_time"] >= 0
    assert got["answer_time"] >= 0


def test_validate_public_url_rejects_localhost() -> None:
    with pytest.raises(SSRFError):
        validate_public_url("http://localhost/secret")


def test_validate_public_url_allows_public_https() -> None:
    assert validate_public_url("https://www.python.org") == "https://www.python.org"


def test_redact_proxy() -> None:
    assert redact_proxy("http://user:pass@proxy:8080") == "http://***:***@proxy:8080"
    assert redact_proxy("http://proxy:8080") == "http://proxy:8080"


def test_proxy_rotator_round_robin() -> None:
    rotator = ProxyRotator(["http://a:1", "http://b:2"])
    assert rotator.next() == "http://a:1"
    assert rotator.next() == "http://b:2"
    assert rotator.next() == "http://a:1"


def test_resolve_text_engines_auto_and_explicit() -> None:
    auto = _resolve_text_engines("auto")
    assert "google" in auto
    assert len(auto) >= 2
    assert _resolve_text_engines("google,mojeek") == ["google", "mojeek"]
    assert _resolve_text_engines("not-a-engine") == []


def test_search_cache_key_stable() -> None:
    a = search_cache_key("ddgs", "query", "wt-wt", 8)
    b = search_cache_key("ddgs", "query", "wt-wt", 8)
    assert a == b
    assert a != search_cache_key("ddgs", "other", "wt-wt", 8)
    assert a != search_cache_key("ddgs", "query", "wt-wt", 8, backend="bing")
    assert search_cache_key("ddgs", "query", None, 8) == search_cache_key(
        "ddgs", "query", "wt-wt", 8
    )


def test_chunk_pages_splits_text() -> None:
    page = ExtractedPage(
        source_id="s1",
        url="https://docs.python.org/3/library/asyncio.html",
        final_url="https://docs.python.org/3/library/asyncio.html",
        title="asyncio — Asynchronous I/O",
        text="Paragraph one about asyncio.\n\nParagraph two about event loops.",
        content_hash="abc",
    )
    passages = chunk_pages([page], passage_tokens=20, overlap_tokens=5)
    assert len(passages) >= 1
    assert all(p.token_count > 0 for p in passages)


def test_chunk_pages_oversized_block_uses_sliding_window() -> None:
    """A single long block must not produce thousands of duplicate passages."""
    text = "asyncio event loop schedules coroutines. " * 400
    page = ExtractedPage(
        source_id="s1",
        url="https://docs.python.org/3/library/asyncio.html",
        final_url="https://docs.python.org/3/library/asyncio.html",
        title="asyncio",
        text=text,
        content_hash="abc",
    )
    passages = chunk_pages([page], passage_tokens=180, overlap_tokens=30)
    assert 2 <= len(passages) <= 50
    assert all(p.token_count <= 180 for p in passages)


def test_bm25_prefers_keyword_match() -> None:
    settings = get_settings()
    passages = [
        Passage(
            passage_id="a#0",
            source_id="a",
            url="https://a",
            title="Asyncio",
            text="The asyncio event loop schedules coroutines.",
            start_char=0,
            end_char=40,
            token_count=8,
        ),
        Passage(
            passage_id="b#0",
            source_id="b",
            url="https://b",
            title="Fruit",
            text="Bananas are yellow.",
            start_char=0,
            end_char=19,
            token_count=4,
        ),
    ]
    results = bm25_search("asyncio event loop", passages, top_k=2, settings=settings)
    assert results[0].passage.passage_id == "a#0"


def test_merge_candidates_combines_sources() -> None:
    settings = get_settings()
    p1 = Passage(
        passage_id="a#0",
        source_id="a",
        url=None,
        title=None,
        text="one",
        start_char=0,
        end_char=3,
        token_count=1,
    )
    p2 = Passage(
        passage_id="b#0",
        source_id="b",
        url=None,
        title=None,
        text="two",
        start_char=0,
        end_char=3,
        token_count=1,
    )
    bm25 = [ScoredPassage(passage=p1, bm25_score=1.0)]
    emb = [ScoredPassage(passage=p2, embedding_score=0.9)]
    merged = merge_candidates(bm25, emb, top_k=2, settings=settings)
    assert len(merged) == 2


def test_page_from_cache_web_fetch_response() -> None:
    cached = {
        "url": "https://www.python.org",
        "final_url": "https://www.python.org",
        "title": "Welcome to Python.org",
        "text": "Python is a programming language that lets you work quickly.",
        "html": "<html><body>raw</body></html>",
        "cleaned_html": "<body>clean</body>",
        "extractor": "crawl4ai",
        "content_hash": "deadbeef",
        "timing_ms": {"fetch": 1, "total": 1},
    }
    page = page_from_cache(cached)
    assert page.url == "https://www.python.org"
    assert page.html == "<html><body>raw</body></html>"
    assert page.cleaned_html == "<body>clean</body>"
    assert page.extractor == "crawl4ai"


def test_debug_dumper_writes_meta_and_files(tmp_path) -> None:
    from pathlib import Path

    from noxa.debug_dump import DebugDumper, write_page_html_files
    from noxa.schemas import ExtractedPage

    dumper = DebugDumper(
        endpoint="/web_search",
        request={"query": "test", "max_results": 3},
        dump_base=Path(tmp_path),
    )
    assert dumper.meta_file.exists()
    stub = dumper.meta_file.read_text(encoding="utf-8")
    assert "Start here" in stub
    assert "in progress" in stub
    dumper.write_json("search_results.json", [{"url": "https://example.org"}], stage="search")
    dumper.add_stage("search", latency_ms=42, files=["search_results.json"])
    page = ExtractedPage(
        source_id="s1",
        url="https://example.org",
        final_url="https://example.org",
        title="Example",
        text="hello",
        content_hash="abc",
        html="<html><body>raw</body></html>",
        cleaned_html="<body>clean</body>",
    )
    html_files = write_page_html_files(dumper, [page])
    out = dumper.finalize(timing={"search": 42, "total": 42})

    assert out.is_dir()
    assert (out / "meta.md").exists()
    meta = (out / "meta.md").read_text(encoding="utf-8")
    assert "Start here" in meta
    assert "Table of contents" in meta
    assert "Complete file index" in meta
    assert "[search_results.json](./search_results.json)" in meta
    assert (out / "pages_html/page_1_raw.html").exists()
    assert (out / "pages_html/page_1_cleaned.html").exists()
    assert "web_search_" in out.name
    assert len(html_files) >= 3


def test_page_cache_payload_includes_html() -> None:
    from noxa.fetcher import page_cache_payload
    from noxa.schemas import ExtractedPage

    page = ExtractedPage(
        source_id="s1",
        url="https://example.org",
        final_url="https://example.org",
        title="T",
        text="markdown",
        content_hash="x",
        html="<html>raw</html>",
        cleaned_html="<body>clean</body>",
    )
    payload = page_cache_payload(page, max_chars=100_000)
    assert payload["html"] == "<html>raw</html>"
    assert payload["cleaned_html"] == "<body>clean</body>"


def test_bm25_matches_punctuation_stripped_words() -> None:
    settings = get_settings()
    passages = [
        Passage(
            passage_id="a#0",
            source_id="a",
            url="https://starmadedock.net/endgoals",
            title="StarMade Endgoals",
            text="The endgoal of StarMade is progression in a sandbox galaxy.",
            start_char=0,
            end_char=60,
            token_count=12,
        ),
        Passage(
            passage_id="b#0",
            source_id="b",
            url="https://b",
            title="Other",
            text="Unrelated content about cooking.",
            start_char=0,
            end_char=30,
            token_count=5,
        ),
    ]
    results = bm25_search("Whats Endgoal?", passages, top_k=2, settings=settings)
    assert results[0].passage.passage_id == "a#0"
    assert results[0].passage.passage_id != results[1].passage.passage_id


def test_fetch_match_result_to_url_by_result_url() -> None:
    from types import SimpleNamespace

    from noxa.fetcher import _match_result_to_url

    results = [
        SimpleNamespace(
            success=True,
            url="https://b.example/page",
            metadata={},
            error_message=None,
        ),
        SimpleNamespace(
            success=True,
            url="https://a.example/page",
            metadata={},
            error_message=None,
        ),
    ]
    used: set[int] = set()
    matched = _match_result_to_url("https://a.example/page", results, used)
    assert matched is results[1]
    assert 1 in used


def test_build_answer_documents_aligns_source_ids() -> None:
    from noxa.pipeline import _build_answer_documents_and_sources
    from noxa.schemas import SelectedPassage

    selected = [
        SelectedPassage(
            source_id="src_a",
            url="https://a",
            title="Source A",
            passage_id="src_a#1",
            text="first passage",
            score=0.9,
        ),
        SelectedPassage(
            source_id="src_a",
            url="https://a",
            title="Source A",
            passage_id="src_a#2",
            text="second passage",
            score=0.8,
        ),
        SelectedPassage(
            source_id="src_b",
            url="https://b",
            title="Source B",
            passage_id="src_b#1",
            text="other source",
            score=0.7,
        ),
    ]
    documents, sources = _build_answer_documents_and_sources(selected)
    assert len(documents) == 2
    assert len(sources) == 2
    assert sources[0].id == 1
    assert sources[1].id == 2
    assert len(sources[0].selected_passages) == 2
    assert "first passage" in documents[0]["text"]
    assert "second passage" not in documents[0]["text"]


def test_build_answer_documents_prefers_top_passage_for_llm_context() -> None:
    from noxa.pipeline import _build_answer_documents_and_sources
    from noxa.schemas import SelectedPassage

    selected = [
        SelectedPassage(
            source_id="review",
            url="https://example.com/review",
            title="Poke Review",
            passage_id="review#verdict",
            text="Reviewers find Poke promising and useful, though not perfect.",
            score=0.86,
        ),
        SelectedPassage(
            source_id="review",
            url="https://example.com/review",
            title="Poke Review",
            passage_id="review#what",
            text="What is Poke? Poke is a proactive AI assistant.",
            score=0.72,
        ),
    ]
    documents, _sources = _build_answer_documents_and_sources(selected)
    assert "Reviewers find Poke promising" in documents[0]["text"]
    assert "What is Poke?" not in documents[0]["text"]


def test_clean_page_markdown_strips_link_targets() -> None:
    from noxa.text_clean import clean_page_markdown

    raw = "# [What is the endgoal?](https://example.com/q)\n[Ask Question](https://x)"
    cleaned = clean_page_markdown(raw)
    assert "https://example.com" not in cleaned
    assert "What is the endgoal?" in cleaned
    assert "Ask Question" in cleaned


def test_rerank_exchange_builds_input_and_output() -> None:
    from noxa.retrieval.rerank import (
        build_rerank_exchange,
        rerank_input_text,
        rerank_output_payload,
    )
    from noxa.schemas import Passage, ScoredPassage

    class _Backend:
        backend_id = "llama_cpp"
        model_id = "rerank.gguf"

    passage = Passage(
        passage_id="p1",
        source_id="s1",
        url="https://example.com",
        title="Example",
        text="Example passage text",
        token_count=3,
        start_char=0,
        end_char=20,
    )
    cand = ScoredPassage(passage=passage, merged_score=0.5)
    exchange = build_rerank_exchange(
        "what is noxa?",
        [cand],
        [0.82],
        _Backend(),
        final_top_k=8,
        max_per_url=2,
    )
    assert exchange["candidate_count"] == 1
    assert exchange["pairs"][0]["score"] == 0.82
    assert "<Query>: what is noxa?" in exchange["pairs"][0]["prompt"]
    assert "Query: what is noxa?" in rerank_input_text(exchange)
    assert rerank_output_payload(exchange)["scores"][0]["score"] == 0.82


def test_resolve_mode_limits_uses_preset_when_null() -> None:
    from noxa.config import MODE_PRESETS, RuntimeMode, resolve_mode_limits

    cfg = MODE_PRESETS[RuntimeMode.DEFAULT]
    assert resolve_mode_limits(cfg, None, None) == (6, 5)
    assert resolve_mode_limits(cfg, 10, None) == (10, 5)
    assert resolve_mode_limits(cfg, None, 2) == (6, 2)


def test_mode_presets_define_search_and_page_defaults() -> None:
    from noxa.config import MODE_PRESETS, RuntimeMode

    assert MODE_PRESETS[RuntimeMode.FAST].max_search_results == 5
    assert MODE_PRESETS[RuntimeMode.FAST].max_pages == 3
    assert MODE_PRESETS[RuntimeMode.FAST].answer_model == "answer_fast"
    assert MODE_PRESETS[RuntimeMode.DEFAULT].max_search_results == 6
    assert MODE_PRESETS[RuntimeMode.DEFAULT].max_pages == 5
    assert MODE_PRESETS[RuntimeMode.DEFAULT].answer_model == "answer_default"
    assert MODE_PRESETS[RuntimeMode.QUALITY].max_search_results == 30
    assert MODE_PRESETS[RuntimeMode.QUALITY].max_pages == 20
    assert MODE_PRESETS[RuntimeMode.QUALITY].answer_model == "answer_default"


def test_compute_confidence_levels() -> None:
    from noxa.confidence import compute_confidence

    assert compute_confidence(
        abstained=True,
        incomplete=False,
        top_rerank_score=0.95,
        has_citations=True,
    ) == "low"
    assert compute_confidence(
        abstained=False,
        incomplete=False,
        top_rerank_score=0.9,
        has_citations=True,
    ) == "high"
    assert compute_confidence(
        abstained=False,
        incomplete=False,
        top_rerank_score=0.7,
        has_citations=False,
    ) == "medium"


def test_write_page_cache_files(tmp_path) -> None:
    from pathlib import Path

    from noxa.debug_dump import DebugDumper, write_page_cache_files

    dumper = DebugDumper(
        endpoint="/web_answer",
        request={"query": "q"},
        dump_base=Path(tmp_path),
    )
    records = [
        {
            "url": "https://cached.example",
            "cache_key": "abc",
            "cache_hit": True,
            "fetched_this_request": False,
            "stored": {
                "url": "https://cached.example",
                "text": "cached text",
                "html": "<html>c</html>",
                "cleaned_html": "<body>c</body>",
            },
        }
    ]
    files = write_page_cache_files(dumper, records)
    assert (dumper.dir / "pages_cache/page_1_raw.html").exists()
    assert (dumper.dir / "pages_cache/page_1_text.md").read_text() == "cached text"
    assert "pages_cache/index.json" in files
