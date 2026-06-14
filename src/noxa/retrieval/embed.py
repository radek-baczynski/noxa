from __future__ import annotations

import time

from noxa.cache import CacheStore
from noxa.config import Settings
from noxa.model_timing import elapsed_ms, log_model
from noxa.runtime.interfaces import EmbedBackend, EmbedKind
from noxa.schemas import Passage, ScoredPassage


def _cache_key(backend: EmbedBackend, kind: EmbedKind, text: str) -> str:
    from noxa.runtime.llama_common import embedding_cache_key

    return embedding_cache_key(backend.backend_id, backend.model_id, f"{kind}: {text}")


class EmbeddingRetriever:
    def __init__(
        self,
        settings: Settings,
        cache: CacheStore,
        backend: EmbedBackend,
    ) -> None:
        self.settings = settings
        self.cache = cache
        self.backend = backend

    async def embed_texts(self, texts: list[str], prefix: str) -> list[list[float]]:
        kind: EmbedKind = "query" if prefix.startswith("query") else "passage"
        keys = [_cache_key(self.backend, kind, t) for t in texts]
        cached = await self.cache.get_embeddings(keys)
        missing_idx = [i for i, k in enumerate(keys) if k not in cached]
        if missing_idx:
            to_embed = [texts[i] for i in missing_idx]
            vectors = self.backend.encode(to_embed, kind)
            new_items = {keys[i]: vectors[j] for j, i in enumerate(missing_idx)}
            await self.cache.set_embeddings(new_items)
            cached.update(new_items)
        return [cached[k] for k in keys]

    async def search(
        self, query: str, passages: list[Passage], top_k: int
    ) -> list[ScoredPassage]:
        if not passages:
            return []
        t_total = time.perf_counter()
        log_model(
            "infer start",
            self.backend.model_id,
            op="embed_search",
            backend=self.backend.backend_id,
            passages=len(passages),
            top_k=top_k,
        )

        passage_keys = [
            _cache_key(self.backend, "passage", p.text) for p in passages
        ]
        query_key = _cache_key(self.backend, "query", query)
        cached_passages = await self.cache.get_embeddings(passage_keys)
        missing = [i for i, k in enumerate(passage_keys) if k not in cached_passages]
        if missing:
            texts = [passages[i].text for i in missing]
            vectors = self.backend.encode(texts, "passage")
            new_items = {
                passage_keys[i]: vectors[j] for j, i in enumerate(missing)
            }
            await self.cache.set_embeddings(new_items)
            cached_passages.update(new_items)
        else:
            log_model(
                "infer cache_hit",
                self.backend.model_id,
                op="embed_passages",
                count=len(passages),
            )

        query_cached = await self.cache.get_embeddings([query_key])
        if query_key not in query_cached:
            qvec = self.backend.encode([query], "query")[0]
            await self.cache.set_embeddings({query_key: qvec})
        else:
            log_model("infer cache_hit", self.backend.model_id, op="embed_query")
            qvec = query_cached[query_key]

        t0 = time.perf_counter()
        knn = await self.cache.knn_embeddings(qvec, passage_keys, top_k)
        knn_ms = elapsed_ms(t0)

        key_to_passage = dict(zip(passage_keys, passages, strict=True))
        results = [
            ScoredPassage(passage=key_to_passage[key], embedding_score=score)
            for key, score in knn
        ]
        log_model(
            "infer done",
            self.backend.model_id,
            elapsed_ms(t_total),
            op="embed_search",
            backend=self.backend.backend_id,
            results=len(results),
            knn_ms=knn_ms,
            encoded_passages=len(missing),
        )
        return results
