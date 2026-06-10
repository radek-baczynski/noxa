from __future__ import annotations

import hashlib
import time
from functools import lru_cache

from noxa.config import Settings
from noxa.model_timing import elapsed_ms, log_model, timed_model_op
from noxa.runtime.interfaces import EmbedKind
def embedding_cache_key(backend_id: str, model_id: str, text: str) -> str:
    raw = f"elastic-v1|{backend_id}|{model_id}|{text}"
    return hashlib.sha256(raw.encode()).hexdigest()


def _prefix_text(text: str, kind: EmbedKind) -> str:
    return f"{kind}: {text}"


@lru_cache(maxsize=1)
def _load_fastembed(model_name: str):
    from fastembed import TextEmbedding

    log_model("load start", model_name, backend="onnx")
    t0 = time.perf_counter()
    model = TextEmbedding(model_name=model_name)
    log_model("load done", model_name, elapsed_ms(t0), backend="onnx")
    return model


class OnnxEmbedBackend:
    backend_id = "onnx"

    def __init__(
        self,
        batch_size: int = 64,
        model_id: str | None = None,
        settings: Settings | None = None,
    ) -> None:
        if model_id is None and settings and settings.embed_model:
            model_id = settings.embed_model
        if model_id is None:
            from noxa.runtime.fastembed_resolve import resolve_fastembed_embed

            model_id = resolve_fastembed_embed()
        self.model_id = model_id
        self.batch_size = batch_size

    def encode(self, texts: list[str], kind: EmbedKind) -> list[list[float]]:
        if not texts:
            return []
        model = _load_fastembed(self.model_id)
        prefixed = [_prefix_text(t, kind) for t in texts]
        with timed_model_op(
            self.model_id,
            "infer encode",
            infer_kind="embedding",
            infer_calls=len(texts),
            backend=self.backend_id,
            count=len(texts),
            kind=kind,
        ):
            vectors = list(model.embed(prefixed, batch_size=self.batch_size))
        return [v.tolist() for v in vectors]
