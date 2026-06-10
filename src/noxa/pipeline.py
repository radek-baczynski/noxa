from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from noxa.model_stats import get_model_infer_stats
from noxa.request_context import log_prefix

from noxa.answerer import Answerer
from noxa.runtime.registry import RuntimeRegistry
from noxa.chunker import chunk_pages
from noxa.confidence import compute_confidence
from noxa.config import RuntimeMode, Settings, resolve_mode_limits
from noxa.debug_dump import (
    DebugDumper,
    page_dict,
    passage_dict,
    scored_passage_dict,
    selected_passage_dict,
    write_page_cache_files,
    write_page_html_files,
)
from noxa.fetcher import PageFetcher, fetch_cache_key, page_cache_payload, page_from_cache
from noxa.packer import pack_context
from noxa.retrieval.bm25 import bm25_search
from noxa.retrieval.embed import EmbeddingRetriever
from noxa.retrieval.merge import merge_candidates
from noxa.retrieval.rerank import Reranker
from noxa.schemas import (
    AnswerPassage,
    AnswerSource,
    ContentSelectResponse,
    DocumentInput,
    ExtractedPage,
    ModelInferMs,
    SelectedPassage,
    SelectionDebug,
    TimingMs,
    WebAnswerResponse,
)
from noxa.search.ddgs import DDGSSearchProvider, search_cache_key

logger = logging.getLogger(__name__)


def _build_answer_documents_and_sources(
    selected: list[SelectedPassage],
) -> tuple[list[dict[str, str]], list[AnswerSource]]:
    """One prompt document per unique source; IDs align with AnswerSource.id."""
    by_source: dict[str, list[SelectedPassage]] = {}
    source_order: list[str] = []
    for sel in selected:
        if sel.source_id not in by_source:
            by_source[sel.source_id] = []
            source_order.append(sel.source_id)
        by_source[sel.source_id].append(sel)

    documents: list[dict[str, str]] = []
    sources: list[AnswerSource] = []
    for source_id, sid in enumerate(source_order, start=1):
        group = by_source[sid]
        first = group[0]
        combined = "\n\n".join(p.text for p in group)
        title = first.title or "Source"
        documents.append({"text": f"{title}: {combined}"})
        sources.append(
            AnswerSource(
                id=source_id,
                url=first.url,
                title=first.title,
                selected_passages=[
                    AnswerPassage(
                        passage_id=p.passage_id,
                        text=p.text,
                        score=p.score,
                    )
                    for p in group
                ],
            )
        )
    return documents, sources


class Pipeline:
    def __init__(
        self,
        settings: Settings,
        cache: Any,
        search: DDGSSearchProvider,
        fetcher: PageFetcher,
        registry: RuntimeRegistry,
    ) -> None:
        self.settings = settings
        self.cache = cache
        self.search = search
        self.fetcher = fetcher
        self.registry = registry
        self.embedder = EmbeddingRetriever(settings, cache, registry.embed)
        self.reranker = Reranker(settings, registry.rerank)

    async def select_from_documents(
        self,
        query: str,
        documents: list[DocumentInput],
        mode: RuntimeMode = RuntimeMode.DEFAULT,
        token_budget: int = 3500,
        return_debug: bool = False,
    ) -> ContentSelectResponse:
        logger.info(
            "%scontent_select start query=%r mode=%s docs=%d",
            log_prefix(),
            query,
            mode,
            len(documents),
        )
        mode_cfg = self.settings.get_mode(mode)
        pages = [
            ExtractedPage(
                source_id=doc.id or f"doc_{i}",
                url=doc.url or "",
                final_url=doc.url or "",
                title=doc.title,
                text=doc.text,
                content_hash="",
            )
            for i, doc in enumerate(documents)
        ]
        passages = chunk_pages(
            pages, mode_cfg.passage_tokens, mode_cfg.overlap_tokens
        )
        selected, debug, trace = await self._select_passages(
            query, passages, mode_cfg, token_budget
        )
        logger.info(
            "%scontent_select done passages=%d selected=%d",
            log_prefix(),
            debug.total_passages,
            debug.final_passages,
        )
        response = ContentSelectResponse(query=query, selected_passages=selected)
        if return_debug:
            response.debug = debug
            dumper = DebugDumper(
                endpoint="/content_select",
                request={
                    "query": query,
                    "mode": mode,
                    "token_budget": token_budget,
                    "documents": [d.model_dump() for d in documents],
                },
                dump_base=self.settings.debug_dump_dir,
            )
            dumper.write_json(
                "input_documents.json",
                [d.model_dump() for d in documents],
                stage="input",
            )
            dumper.write_json(
                "pages.json", [page_dict(p) for p in pages], stage="chunk"
            )
            dumper.write_json(
                "passages_all.json",
                [passage_dict(p) for p in passages],
                stage="chunk",
            )
            dumper.add_stage(
                "chunk",
                notes=f"{len(pages)} pages → {len(passages)} passages",
                files=["pages.json", "passages_all.json"],
            )
            dumper.write_json("retrieval_bm25.json", trace["bm25"], stage="select")
            dumper.write_json("retrieval_embed.json", trace["embed"], stage="select")
            dumper.write_json("retrieval_merged.json", trace["merged"], stage="select")
            dumper.write_json("retrieval_reranked.json", trace["reranked"], stage="select")
            dumper.write_json(
                "selected_passages.json",
                [selected_passage_dict(s) for s in selected],
                stage="select",
            )
            dumper.add_stage(
                "select",
                notes=(
                    f"bm25={debug.bm25_candidates} embed={debug.embedding_candidates} "
                    f"merged={debug.merged_candidates} rerank={debug.reranked_candidates} "
                    f"final={debug.final_passages}"
                ),
                files=[
                    "retrieval_bm25.json",
                    "retrieval_embed.json",
                    "retrieval_merged.json",
                    "retrieval_reranked.json",
                    "selected_passages.json",
                ],
            )
            model_infer = get_model_infer_stats().to_dict()
            dump_path = dumper.finalize(
                mode_config=mode_cfg.model_dump(),
                model_infer=model_infer,
                response=response.model_dump(),
            )
            response.debug = {
                **debug.model_dump(),
                "model_infer": model_infer,
                "dump_dir": str(dump_path),
                "meta_md": str(dumper.meta_file),
            }
        return response

    async def _select_passages(
        self,
        query: str,
        passages: list,
        mode_cfg,
        token_budget: int,
    ) -> tuple[list[SelectedPassage], SelectionDebug, dict[str, list]]:
        logger.debug(
            "%sretrieval start passages=%d top_k bm25=%d embed=%d",
            log_prefix(),
            len(passages),
            mode_cfg.bm25_top_k,
            mode_cfg.embedding_top_k,
        )
        t0 = time.perf_counter()
        bm25, emb = await asyncio.gather(
            asyncio.to_thread(
                bm25_search, query, passages, mode_cfg.bm25_top_k, self.settings
            ),
            self.embedder.search(query, passages, mode_cfg.embedding_top_k),
        )
        select_ms = int((time.perf_counter() - t0) * 1000)
        logger.info(
            "%sretrieval bm25+embed done bm25=%d embed=%d latency=%dms",
            log_prefix(),
            len(bm25),
            len(emb),
            select_ms,
        )
        merged = merge_candidates(
            bm25, emb, mode_cfg.merged_top_k, self.settings
        )
        logger.info(
            "%sretrieval merge done candidates=%d",
            log_prefix(),
            len(merged),
        )
        t0 = time.perf_counter()
        reranked = await asyncio.to_thread(
            self.reranker.rerank,
            query,
            merged,
            mode_cfg.rerank_final_top_k,
        )
        logger.info(
            "%sretrieval rerank done selected=%d latency=%dms",
            log_prefix(),
            len(reranked),
            int((time.perf_counter() - t0) * 1000),
        )
        t0 = time.perf_counter()
        _, kept = pack_context(query, reranked, token_budget)
        logger.info(
            "%sretrieval pack done kept=%d latency=%dms",
            log_prefix(),
            len(kept),
            int((time.perf_counter() - t0) * 1000),
        )
        selected = [
            SelectedPassage(
                source_id=item.passage.source_id,
                url=item.passage.url,
                title=item.passage.title,
                passage_id=item.passage.passage_id,
                text=item.passage.text,
                score=item.rerank_score or 0.0,
            )
            for _, item in kept
        ]
        debug = SelectionDebug(
            total_passages=len(passages),
            bm25_candidates=len(bm25),
            embedding_candidates=len(emb),
            merged_candidates=len(merged),
            reranked_candidates=len(reranked),
            final_passages=len(selected),
        )
        trace = {
            "bm25": [scored_passage_dict(x) for x in bm25],
            "embed": [scored_passage_dict(x) for x in emb],
            "merged": [scored_passage_dict(x) for x in merged],
            "reranked": [scored_passage_dict(x) for x in reranked],
        }
        return selected, debug, trace

    async def web_answer(
        self,
        query: str,
        mode: RuntimeMode = RuntimeMode.DEFAULT,
        max_search_results: int | None = None,
        max_pages: int | None = None,
        return_sources: bool = True,
        return_debug: bool = False,
    ) -> WebAnswerResponse:
        t_total = time.perf_counter()
        mode_cfg = self.settings.get_mode(mode)
        max_results, max_pages = resolve_mode_limits(
            mode_cfg, max_search_results, max_pages
        )
        timing = TimingMs()

        logger.info(
            "%sweb_answer start query=%r mode=%s max_results=%d max_pages=%d",
            log_prefix(),
            query,
            mode,
            max_results,
            max_pages,
        )

        t0 = time.perf_counter()
        skey = search_cache_key("ddgs", query, "wt-wt", max_results)
        cached_search = await self.cache.get(skey)
        if cached_search:
            from noxa.schemas import SearchResultItem

            search_results = [
                SearchResultItem(**r) if isinstance(r, dict) else r
                for r in cached_search
            ]
            logger.info(
                "%sweb_answer search cache_hit results=%d",
                log_prefix(),
                len(search_results),
            )
        else:
            logger.info("%sweb_answer search calling ddgs", log_prefix())
            search_results = await self.search.search(query, max_results)
            await self.cache.set(
                skey,
                [r.model_dump() for r in search_results],
                self.settings.search_ttl_seconds,
            )
            logger.info(
                "%sweb_answer search done results=%d",
                log_prefix(),
                len(search_results),
            )
        timing.search = int((time.perf_counter() - t0) * 1000)

        urls = [r.url for r in search_results[:max_pages] if r.url]

        t0 = time.perf_counter()
        pages: list[ExtractedPage] = []
        failed: list[str] = []
        uncached_urls: list[str] = []
        fetch_cache_records: list[dict[str, Any]] = []
        for url in urls:
            key = fetch_cache_key(url)
            cached = await self.cache.get(key)
            if cached:
                fetch_cache_records.append(
                    {
                        "url": url,
                        "cache_key": key,
                        "cache_hit": True,
                        "fetched_this_request": False,
                        "stored": cached,
                    }
                )
                pages.append(page_from_cache(cached))
            else:
                fetch_cache_records.append(
                    {
                        "url": url,
                        "cache_key": key,
                        "cache_hit": False,
                        "fetched_this_request": False,
                        "stored": None,
                    }
                )
                uncached_urls.append(url)

        if uncached_urls:
            logger.info(
                "%sweb_answer fetch start urls=%d cached=%d",
                log_prefix(),
                len(uncached_urls),
                len(pages),
            )
            fetched, fetch_failed = await self.fetcher.fetch_many(
                uncached_urls, timeout_ms=mode_cfg.per_page_timeout_ms
            )
            failed.extend(fetch_failed)
            for page in fetched:
                payload = page_cache_payload(page, self.settings.max_chars_per_page)
                await self.cache.set(
                    fetch_cache_key(page.url),
                    payload,
                    self.settings.fetch_ttl_seconds,
                )
                for rec in fetch_cache_records:
                    if rec["url"] == page.url:
                        rec["stored"] = payload
                        rec["fetched_this_request"] = True
                        break
                pages.append(page)
            logger.info(
                "%sweb_answer fetch done ok=%d failed=%d",
                log_prefix(),
                len(fetched),
                len(fetch_failed),
            )
        else:
            logger.info(
                "%sweb_answer fetch all cached pages=%d",
                log_prefix(),
                len(pages),
            )

        for page in pages:
            logger.info(
                "%sweb_answer page url=%s chars=%d title=%r",
                log_prefix(),
                page.url,
                len(page.text),
                (page.title or "")[:80],
            )

        url_order = {url: i for i, url in enumerate(urls)}
        pages.sort(key=lambda p: url_order.get(p.url, 999))
        timing.fetch = int((time.perf_counter() - t0) * 1000)
        total_chars = sum(len(p.text) for p in pages)
        logger.info(
            "%sweb_answer pages ready count=%d total_chars=%d fetch_ms=%d",
            log_prefix(),
            len(pages),
            total_chars,
            timing.fetch,
        )

        if not pages:
            raise RuntimeError("Search returned results, but no pages could be fetched.")

        for i, page in enumerate(pages):
            page.source_rank = i + 1

        logger.info(
            "%sweb_answer chunk start passage_tokens=%d overlap=%d",
            log_prefix(),
            mode_cfg.passage_tokens,
            mode_cfg.overlap_tokens,
        )
        t_chunk = time.perf_counter()
        passages = chunk_pages(
            pages, mode_cfg.passage_tokens, mode_cfg.overlap_tokens
        )
        logger.info(
            "%sweb_answer chunk done pages=%d passages=%d latency=%dms",
            log_prefix(),
            len(pages),
            len(passages),
            int((time.perf_counter() - t_chunk) * 1000),
        )

        t0 = time.perf_counter()
        logger.info(
            "%sweb_answer select start passages=%d token_budget=%d",
            log_prefix(),
            len(passages),
            mode_cfg.max_context_tokens,
        )
        selected, debug, trace = await self._select_passages(
            query, passages, mode_cfg, mode_cfg.max_context_tokens
        )
        timing.select = int((time.perf_counter() - t0) * 1000)
        logger.info(
            "%sweb_answer select done selected=%d latency=%dms",
            log_prefix(),
            len(selected),
            timing.select,
        )

        documents, sources = _build_answer_documents_and_sources(selected)
        doc_chars = sum(len(d["text"]) for d in documents)
        top_rerank_score = selected[0].score if selected else None
        answer_role = self.settings.answer_role_for_mode(mode)
        answerer = Answerer(self.registry, answer_role)
        backend = answerer.backend
        logger.info(
            "%sweb_answer generate start role=%s backend=%s model=%s passages=%d doc_chars=%d max_output=%d",
            log_prefix(),
            answer_role,
            backend.backend_id,
            backend.model_id,
            len(documents),
            doc_chars,
            mode_cfg.max_output_tokens,
        )
        t0 = time.perf_counter()
        answer_result = await asyncio.to_thread(
            answerer.answer,
            query,
            documents,
            mode_cfg.max_output_tokens,
        )
        answer_text = answer_result.answer
        abstained = answer_result.abstained
        citations = answer_result.citations
        if answer_result.incomplete:
            logger.warning(
                "%sweb_answer answer incomplete status=%r output may be truncated",
                log_prefix(),
                answer_result.status,
            )
        timing.answer = int((time.perf_counter() - t0) * 1000)
        timing.total = int((time.perf_counter() - t_total) * 1000)
        perf_msg = ""
        if answer_result.llm_perf:
            p = answer_result.llm_perf
            perf_msg = (
                f" prefill={p.prefill_tps}tok/s decode={p.decode_tps}tok/s"
                f" ({p.prefill_ms}ms+{p.decode_ms}ms)"
            )
        logger.info(
            "%sweb_answer done abstained=%s citations=%d total=%dms "
            "(search=%d fetch=%d select=%d answer=%d)%s",
            log_prefix(),
            abstained,
            len(citations),
            timing.total,
            timing.search or 0,
            timing.fetch or 0,
            timing.select or 0,
            timing.answer or 0,
            perf_msg,
        )

        if not return_sources:
            sources = []

        for c in citations:
            for s in sources:
                if c.source_id == s.id:
                    c.url = s.url
                    c.title = s.title

        confidence = compute_confidence(
            abstained=abstained,
            incomplete=answer_result.incomplete,
            top_rerank_score=top_rerank_score,
            has_citations=bool(citations),
            model_confidence=answer_result.confidence,
        )

        model_infer = ModelInferMs(**get_model_infer_stats().to_dict())
        resp = WebAnswerResponse(
            query=query,
            answer=answer_text,
            confidence=confidence,
            abstained=abstained,
            citations=citations,
            sources=sources,
            timing_ms=timing,
            model_infer=model_infer,
        )
        if return_debug:
            dumper = DebugDumper(
                endpoint="/web_answer",
                request={
                    "query": query,
                    "mode": mode,
                    "max_search_results": max_results,
                    "max_pages": max_pages,
                    "return_sources": return_sources,
                },
                dump_base=self.settings.debug_dump_dir,
            )
            dumper.write_json(
                "search_results.json",
                [r.model_dump() for r in search_results],
                stage="search",
            )
            dumper.add_stage(
                "search",
                latency_ms=timing.search,
                notes=f"{len(search_results)} results",
                files=["search_results.json"],
            )
            dumper.write_json("fetch_urls.json", {"urls": urls, "failed": failed}, stage="fetch")
            cache_files = write_page_cache_files(dumper, fetch_cache_records, stage="fetch")
            page_sources = [
                "cache_hit" if r["cache_hit"] and not r.get("fetched_this_request") else "fetched"
                for r in sorted(
                    fetch_cache_records,
                    key=lambda r: urls.index(r["url"]) if r["url"] in urls else 999,
                )
            ]
            dumper.write_json(
                "pages_raw.json", [page_dict(p) for p in pages], stage="fetch"
            )
            html_files = write_page_html_files(
                dumper, pages, stage="fetch", sources=page_sources
            )
            for i, page in enumerate(pages):
                dumper.write_text(
                    f"pages_raw/page_{i + 1}.md",
                    f"# {page.title or page.url}\n\nURL: {page.url}\n\n{page.text}",
                    stage="fetch",
                )
            dumper.add_stage(
                "fetch",
                latency_ms=timing.fetch,
                notes=(
                    f"{len(pages)} pages, {len(failed)} failed, {total_chars} chars; "
                    f"{sum(1 for r in fetch_cache_records if r['cache_hit'] and not r.get('fetched_this_request'))} cache hits"
                ),
                files=["fetch_urls.json", "pages_cache/index.json", "pages_raw.json"]
                + cache_files
                + html_files
                + [f"pages_raw/page_{i + 1}.md" for i in range(len(pages))],
            )
            dumper.write_json(
                "passages_all.json",
                [passage_dict(p) for p in passages],
                stage="chunk",
            )
            dumper.add_stage(
                "chunk",
                notes=f"{len(passages)} passages from {len(pages)} pages",
                files=["passages_all.json"],
            )
            dumper.write_json("retrieval_bm25.json", trace["bm25"], stage="select")
            dumper.write_json("retrieval_embed.json", trace["embed"], stage="select")
            dumper.write_json("retrieval_merged.json", trace["merged"], stage="select")
            dumper.write_json("retrieval_reranked.json", trace["reranked"], stage="select")
            dumper.write_json(
                "selected_passages.json",
                [selected_passage_dict(s) for s in selected],
                stage="select",
            )
            dumper.add_stage(
                "select",
                latency_ms=timing.select,
                notes=debug.model_dump(),
                files=[
                    "retrieval_bm25.json",
                    "retrieval_embed.json",
                    "retrieval_merged.json",
                    "retrieval_reranked.json",
                    "selected_passages.json",
                ],
            )
            dumper.write_json("answer_documents.json", documents, stage="answer")
            dumper.write_text(
                "model_prompt.txt", answer_result.model_prompt, stage="answer"
            )
            dumper.write_text(
                "model_raw.txt", answer_result.raw_response, stage="answer"
            )
            parsed_payload: dict[str, object] = {
                "answer": answer_result.answer,
                "abstained": answer_result.abstained,
                "incomplete": answer_result.incomplete,
                "status": answer_result.status,
                "confidence": answer_result.confidence,
                "backend": answer_result.backend,
            }
            if answer_result.llm_perf:
                parsed_payload["llm_perf"] = answer_result.llm_perf.to_dict()
            dumper.write_json(
                "model_parsed.json",
                parsed_payload,
                stage="answer",
            )
            dumper.add_stage(
                "answer",
                latency_ms=timing.answer,
                notes=f"role={answer_role} backend={backend.backend_id} model={backend.model_id}",
                files=[
                    "answer_documents.json",
                    "model_prompt.txt",
                    "model_raw.txt",
                    "model_parsed.json",
                ],
            )
            dump_path = dumper.finalize(
                mode_config=mode_cfg.model_dump(),
                timing=timing.model_dump(),
                model_infer=model_infer.model_dump(),
                response=resp.model_dump(),
                extra_meta={
                    "search_cache_key": skey,
                    "answer_role": answer_role,
                    "answer_backend": backend.backend_id,
                    "answer_model": backend.model_id,
                },
            )
            resp.debug = {
                "failed_urls": failed,
                "selection": debug.model_dump(),
                "model_infer": model_infer.model_dump(),
                "dump_dir": str(dump_path),
                "meta_md": str(dumper.meta_file),
                "answer": {
                    "backend": answer_result.backend,
                    "incomplete": answer_result.incomplete,
                    "status": answer_result.status,
                    "model_confidence": answer_result.confidence,
                    "llm_perf": (
                        answer_result.llm_perf.to_dict()
                        if answer_result.llm_perf
                        else None
                    ),
                },
            }
        return resp
