#!/usr/bin/env python3
"""Smoke tests for Noxa v0 endpoints."""

from __future__ import annotations

import json
import logging
import sys
import time
import uuid
import urllib.error
import urllib.request

BASE = "http://127.0.0.1:8000"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-5s [smoke] %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
    force=True,
)
log = logging.getLogger("smoke")


def _summarize_body(body: dict) -> str:
    parts: list[str] = []
    for key, value in body.items():
        if key == "documents" and isinstance(value, list):
            parts.append(f"documents={len(value)}")
        elif key == "urls" and isinstance(value, list):
            parts.append(f"urls={value!r}")
        elif isinstance(value, str) and len(value) > 80:
            parts.append(f"{key}={value[:77]!r}...")
        else:
            parts.append(f"{key}={value!r}")
    return "{" + ", ".join(parts) + "}"


def post(path: str, body: dict, timeout: int = 120) -> dict:
    request_id = str(uuid.uuid4())
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        f"{BASE}{path}",
        data=data,
        headers={
            "Content-Type": "application/json",
            "X-Request-Id": request_id,
        },
        method="POST",
    )
    log.info("→ POST %s [%s] %s", path, request_id, _summarize_body(body))
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
            elapsed_ms = int((time.perf_counter() - t0) * 1000)
            server_rid = resp.headers.get("x-request-id", request_id)
            log.info(
                "← POST %s [%s] status=%d latency=%dms bytes=%d",
                path,
                server_rid,
                resp.status,
                elapsed_ms,
                len(raw),
            )
            return json.loads(raw)
    except urllib.error.HTTPError as exc:
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        err_body = exc.read().decode("utf-8", errors="replace")
        log.error(
            "← POST %s [%s] status=%d latency=%dms body=%s",
            path,
            request_id,
            exc.code,
            elapsed_ms,
            err_body[:500],
        )
        raise RuntimeError(f"HTTP {exc.code}: {err_body[:300]}") from exc
    except urllib.error.URLError as exc:
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        log.error(
            "← POST %s [%s] connection failed latency=%dms: %s",
            path,
            request_id,
            elapsed_ms,
            exc.reason,
        )
        raise


def _log_search_response(r: dict) -> None:
    log.info(
        "  search: %d results in %dms",
        len(r.get("results", [])),
        r.get("timing_ms", {}).get("search", 0),
    )
    for item in r.get("results", [])[:3]:
        log.info("    #%s %s", item.get("rank"), item.get("url"))


def _log_fetch_response(r: dict) -> None:
    log.info(
        "  fetch: title=%r chars=%d total=%dms",
        r.get("title"),
        len(r.get("text", "")),
        r.get("timing_ms", {}).get("total", 0),
    )
    preview = (r.get("text") or "")[:120].replace("\n", " ")
    if preview:
        log.info("    preview: %s...", preview)


def _log_crawl_response(r: dict) -> None:
    timing = r.get("timing_ms", {})
    log.info(
        "  crawl: %d pages, %d failed, %dms",
        len(r.get("pages", [])),
        len(r.get("failed_urls", [])),
        timing.get("crawl", timing.get("total", 0)),
    )
    for page in r.get("pages", [])[:3]:
        log.info(
            "    depth=%s score=%.2f %s",
            page.get("depth"),
            page.get("score", 0),
            page.get("url"),
        )


def _log_select_response(r: dict) -> None:
    debug = r.get("debug") or {}
    log.info("  select: %d passages chosen", len(r.get("selected_passages", [])))
    if debug:
        log.info(
            "    pipeline: passages=%s bm25=%s embed=%s merged=%s rerank=%s final=%s",
            debug.get("total_passages"),
            debug.get("bm25_candidates"),
            debug.get("embedding_candidates"),
            debug.get("merged_candidates"),
            debug.get("reranked_candidates"),
            debug.get("final_passages"),
        )
    for passage in r.get("selected_passages", [])[:2]:
        text = (passage.get("text") or "")[:80].replace("\n", " ")
        log.info("    score=%.3f %s", passage.get("score", 0), text)


def _log_answer_response(r: dict) -> None:
    timing = r.get("timing_ms", {})
    log.info(
        "  answer: abstained=%s citations=%d sources=%d",
        r.get("abstained"),
        len(r.get("citations", [])),
        len(r.get("sources", [])),
    )
    log.info(
        "    timing: total=%dms search=%d fetch=%d select=%d answer=%d",
        timing.get("total", 0),
        timing.get("search", 0),
        timing.get("fetch", 0),
        timing.get("select", 0),
        timing.get("answer", 0),
    )
    answer = (r.get("answer") or "")[:200].replace("\n", " ")
    if answer:
        log.info("    answer: %s...", answer)
    debug = r.get("debug") or {}
    failed = debug.get("failed_urls") or []
    if failed:
        log.warning("    failed_urls: %s", failed)


def run_step(name: str, fn) -> bool:
    log.info("--- %s ---", name)
    t0 = time.perf_counter()
    try:
        fn()
        log.info("OK %s (%dms)", name, int((time.perf_counter() - t0) * 1000))
        return True
    except Exception as exc:
        log.error("FAIL %s (%dms): %s", name, int((time.perf_counter() - t0) * 1000), exc)
        return False


def step_web_search() -> None:
    r = post("/web_search", {"query": "python asyncio", "max_results": 3})
    _log_search_response(r)
    assert r["results"], "expected search results"


def step_web_fetch() -> None:
    r = post("/web_fetch", {"url": "https://www.python.org"}, timeout=60)
    _log_fetch_response(r)
    assert r["text"], "expected markdown text"


def step_web_crawl() -> None:
    r = post(
        "/web_crawl",
        {
            "urls": ["https://docs.python.org/3/"],
            "query": "asyncio",
            "max_pages": 2,
            "max_depth": 1,
        },
        timeout=120,
    )
    _log_crawl_response(r)


def step_content_select() -> None:
    r = post(
        "/content_select",
        {
            "query": "asyncio event loop",
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
            "return_debug": True,
        },
        timeout=300,
    )
    _log_select_response(r)
    assert r["selected_passages"], "expected selected passages"


def step_web_answer() -> None:
    r = post(
        "/web_answer",
        {
            "query": "What is Python asyncio?",
            "mode": "fast",
            "max_search_results": 3,
            "max_pages": 2,
            "return_debug": True,
        },
        timeout=600,
    )
    _log_answer_response(r)
    assert r["answer"], "expected answer text"


def main() -> int:
    log.info("Smoke test starting base=%s", BASE)
    steps = [
        ("1. /web_search", step_web_search),
        ("2. /web_fetch", step_web_fetch),
        ("3. /web_crawl", step_web_crawl),
        ("4. /content_select", step_content_select),
        ("5. /web_answer (fast mode, limited pages)", step_web_answer),
    ]
    failures = sum(1 for name, fn in steps if not run_step(name, fn))

    if failures:
        log.error("Done: %d test(s) failed", failures)
        return 1
    log.info("Done: all smoke tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
