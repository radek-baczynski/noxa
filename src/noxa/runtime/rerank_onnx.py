from __future__ import annotations

import time
from functools import lru_cache

from noxa.config import Settings
from noxa.model_timing import elapsed_ms, log_model, timed_model_op
RERANK_PASSAGE_MAX_CHARS = 512


def _truncate_passage(text: str, max_chars: int = RERANK_PASSAGE_MAX_CHARS) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "..."


@lru_cache(maxsize=1)
def _load_cross_encoder(model_name: str):
    from fastembed.rerank.cross_encoder import TextCrossEncoder

    log_model("load start", model_name, backend="onnx")
    t0 = time.perf_counter()
    model = TextCrossEncoder(model_name=model_name)
    log_model("load done", model_name, elapsed_ms(t0), backend="onnx")
    return model


class OnnxRerankBackend:
    backend_id = "onnx"

    def __init__(
        self,
        model_id: str | None = None,
        settings: Settings | None = None,
    ) -> None:
        if model_id is None and settings and settings.rerank_model:
            model_id = settings.rerank_model
        if model_id is None:
            from noxa.runtime.fastembed_resolve import resolve_fastembed_rerank

            model_id = resolve_fastembed_rerank()
        self.model_id = model_id

    def score(self, query: str, passages: list[str]) -> list[float]:
        if not passages:
            return []
        model = _load_cross_encoder(self.model_id)
        truncated = [_truncate_passage(p) for p in passages]
        with timed_model_op(
            self.model_id,
            "infer rerank",
            infer_kind="rerank",
            infer_calls=len(passages),
            backend=self.backend_id,
            batch=len(passages),
        ):
            scores = list(model.rerank(query, truncated))
        return [float(s) for s in scores]
