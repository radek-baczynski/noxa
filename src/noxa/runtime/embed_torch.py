from __future__ import annotations

import hashlib
import time
from functools import lru_cache

from noxa.model_timing import elapsed_ms, log_model, timed_model_op
from noxa.runtime.detect import infer_device
from noxa.runtime.interfaces import EmbedKind
from noxa.config import Settings
from noxa.runtime.manifest import resolve_artifact


def embedding_cache_key(backend_id: str, model_id: str, text: str) -> str:
    raw = f"elastic-v1|{backend_id}|{model_id}|{text}"
    return hashlib.sha256(raw.encode()).hexdigest()


def _prefix_text(text: str, kind: EmbedKind) -> str:
    return f"{kind}: {text}"


@lru_cache(maxsize=1)
def _load_st_model(model_name: str):
    from sentence_transformers import SentenceTransformer

    device = infer_device()
    log_model("load start", model_name, backend="torch", device=device)
    t0 = time.perf_counter()
    model = SentenceTransformer(model_name, device=device)
    log_model("load done", model_name, elapsed_ms(t0), backend="torch", device=device)
    return model


class TorchEmbedBackend:
    backend_id = "torch"

    def __init__(self, batch_size: int = 64, settings: Settings | None = None) -> None:
        self.model_id = resolve_artifact("embed", "model", settings)
        self.batch_size = batch_size

    def encode(self, texts: list[str], kind: EmbedKind) -> list[list[float]]:
        if not texts:
            return []
        model = _load_st_model(self.model_id)
        prefixed = [_prefix_text(t, kind) for t in texts]
        with timed_model_op(
            self.model_id,
            "infer encode",
            infer_kind="embedding",
            infer_calls=len(texts),
            backend=self.backend_id,
            count=len(texts),
            kind=kind,
            device=str(model.device),
        ):
            vectors = model.encode(
                prefixed,
                batch_size=self.batch_size,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        return [v.tolist() for v in vectors]
