from __future__ import annotations

import time
from functools import lru_cache
from pathlib import Path
from typing import Literal

from noxa.model_timing import elapsed_ms, log_model
from noxa.runtime.artifacts import GgufArtifact
from noxa.runtime.detect import LlamaOffloadConfig

LlamaMode = Literal["generate", "embed", "rerank"]


def resolve_gguf_path(
    artifact: GgufArtifact,
    cache_dir: Path,
    *,
    token: str | None = None,
) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    local = cache_dir / artifact.filename
    if local.exists():
        return local
    from huggingface_hub import hf_hub_download

    downloaded = hf_hub_download(
        repo_id=artifact.repo_id,
        filename=artifact.filename,
        local_dir=str(cache_dir),
        token=token,
    )
    return Path(downloaded)


def _llama_init_kwargs(mode: LlamaMode) -> dict[str, object]:
    import llama_cpp

    if mode == "embed":
        return {"embedding": True}
    # rerank only needs last-token logits; logits_all would compute and store
    # full-vocab logits for every position, which dominates rerank latency.
    return {}


@lru_cache(maxsize=16)
def load_llama(
    repo_id: str,
    filename: str,
    offload_key: str,
    cache_dir: str,
    token: str | None,
    mode: LlamaMode,
) -> tuple[object, str]:
    from llama_cpp import Llama

    artifact = GgufArtifact(repo_id=repo_id, filename=filename)
    model_path = resolve_gguf_path(artifact, Path(cache_dir), token=token)

    parts = offload_key.split(",")
    n_gpu_layers = int(parts[0])
    n_threads = int(parts[1])
    n_ctx = int(parts[2]) if len(parts) > 2 else 4096

    log_model(
        "load start",
        str(model_path),
        backend="llama_cpp",
        repo=repo_id,
        mode=mode,
    )
    t0 = time.perf_counter()
    llm = Llama(
        model_path=str(model_path),
        n_ctx=n_ctx,
        n_gpu_layers=n_gpu_layers,
        n_threads=n_threads,
        verbose=False,
        **_llama_init_kwargs(mode),
    )
    log_model(
        "load done",
        filename,
        elapsed_ms(t0),
        backend="llama_cpp",
        mode=mode,
    )
    return llm, filename


def offload_key(offload: LlamaOffloadConfig) -> str:
    return f"{offload.n_gpu_layers},{offload.n_threads},{offload.n_ctx}"


def embedding_cache_key(backend_id: str, model_id: str, text: str) -> str:
    import hashlib

    raw = f"llama-v1|{backend_id}|{model_id}|{text}"
    return hashlib.sha256(raw.encode()).hexdigest()
