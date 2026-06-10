from __future__ import annotations

import time

from noxa.config import Settings
from noxa.model_timing import elapsed_ms, log_model
from noxa.runtime.interfaces import RerankBackend
from noxa.schemas import ScoredPassage


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
    ) -> list[ScoredPassage]:
        if not candidates:
            return []
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

        passages = [c.passage.text for c in candidates]
        scores = self.backend.score(query, passages)

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
        return selected
