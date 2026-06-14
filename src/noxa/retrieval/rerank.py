from __future__ import annotations

import time
from typing import Any

from noxa.config import Settings
from noxa.model_timing import elapsed_ms, log_model
from noxa.runtime.interfaces import RerankBackend
from noxa.runtime.rerank_llama import format_rerank_document, format_rerank_prompt
from noxa.schemas import ScoredPassage


def build_rerank_exchange(
    query: str,
    candidates: list[ScoredPassage],
    scores: list[float],
    backend: RerankBackend,
    *,
    final_top_k: int,
    max_per_url: int,
) -> dict[str, Any]:
    pairs: list[dict[str, Any]] = []
    for i, (cand, score) in enumerate(zip(candidates, scores, strict=True)):
        passage_text = cand.passage.text
        truncated = format_rerank_document(
            passage_text, title=cand.passage.title, url=cand.passage.url
        )
        pairs.append(
            {
                "index": i,
                "passage_id": cand.passage.passage_id,
                "source_id": cand.passage.source_id,
                "url": cand.passage.url,
                "title": cand.passage.title,
                "passage_text": passage_text,
                "truncated_passage": truncated,
                "prompt": format_rerank_prompt(
                    query,
                    passage_text,
                    title=cand.passage.title,
                    url=cand.passage.url,
                ),
                "score": float(score),
                "bm25_score": cand.bm25_score,
                "embedding_score": cand.embedding_score,
                "merged_score": cand.merged_score,
            }
        )
    pairs.sort(key=lambda item: item["score"], reverse=True)
    return {
        "query": query,
        "backend": backend.backend_id,
        "model": backend.model_id,
        "final_top_k": final_top_k,
        "max_per_url": max_per_url,
        "candidate_count": len(candidates),
        "pairs": pairs,
    }


def rerank_input_text(exchange: dict[str, Any]) -> str:
    lines = [
        f"Query: {exchange['query']}",
        f"Model: {exchange.get('model', '')} ({exchange.get('backend', '')})",
        f"Candidates: {exchange.get('candidate_count', 0)}",
        "",
    ]
    for pair in exchange.get("pairs", []):
        lines.extend(
            [
                f"=== Pair {pair['index']} score={pair['score']} ===",
                f"source_id={pair.get('source_id')} url={pair.get('url')}",
                pair["prompt"],
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def rerank_output_payload(exchange: dict[str, Any]) -> dict[str, Any]:
    return {
        "query": exchange["query"],
        "model": exchange.get("model"),
        "backend": exchange.get("backend"),
        "scores": [
            {
                "index": pair["index"],
                "score": pair["score"],
                "passage_id": pair.get("passage_id"),
                "source_id": pair.get("source_id"),
                "url": pair.get("url"),
                "title": pair.get("title"),
            }
            for pair in exchange.get("pairs", [])
        ],
    }


class Reranker:
    def __init__(self, settings: Settings, backend: RerankBackend) -> None:
        self.settings = settings
        self.backend = backend

    def rerank(
        self,
        query: str,
        candidates: list[ScoredPassage],
        final_top_k: int,
        max_per_url: int | None = None,
    ) -> tuple[list[ScoredPassage], dict[str, Any]]:
        if not candidates:
            return [], {
                "query": query,
                "backend": self.backend.backend_id,
                "model": self.backend.model_id,
                "final_top_k": final_top_k,
                "max_per_url": max_per_url or self.settings.rerank_max_per_url,
                "candidate_count": 0,
                "pairs": [],
            }
        max_per_url = max_per_url or self.settings.rerank_max_per_url
        t_total = time.perf_counter()
        log_model(
            "infer start",
            self.backend.model_id,
            op="rerank",
            backend=self.backend.backend_id,
            candidates=len(candidates),
            final_top_k=final_top_k,
        )

        documents = [
            format_rerank_document(
                c.passage.text, title=c.passage.title, url=c.passage.url
            )
            for c in candidates
        ]
        scores = self.backend.score(query, documents)

        reranked: list[ScoredPassage] = []
        for cand, score in zip(candidates, scores, strict=True):
            reranked.append(
                ScoredPassage(
                    passage=cand.passage,
                    bm25_score=cand.bm25_score,
                    embedding_score=cand.embedding_score,
                    merged_score=cand.merged_score,
                    rerank_score=float(score),
                )
            )
        reranked.sort(key=lambda s: s.rerank_score or 0.0, reverse=True)

        selected: list[ScoredPassage] = []
        per_url: dict[str, int] = {}
        seen_text: set[str] = set()
        for item in reranked:
            url = item.passage.url or item.passage.source_id
            if per_url.get(url, 0) >= max_per_url:
                continue
            snippet_key = item.passage.text[:200]
            if snippet_key in seen_text:
                continue
            seen_text.add(snippet_key)
            per_url[url] = per_url.get(url, 0) + 1
            selected.append(item)
            if len(selected) >= final_top_k:
                break

        log_model(
            "infer done",
            self.backend.model_id,
            elapsed_ms(t_total),
            op="rerank",
            backend=self.backend.backend_id,
            selected=len(selected),
        )
        exchange = build_rerank_exchange(
            query,
            candidates,
            scores,
            self.backend,
            final_top_k=final_top_k,
            max_per_url=max_per_url,
        )
        return selected, exchange
