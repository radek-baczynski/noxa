from __future__ import annotations

import asyncio
import logging
import os
import time
from pathlib import Path

from noxa.config import Settings
from noxa.ml_deps import MLDependencyError, require_ml
from noxa.model_timing import elapsed_ms, log_model
from noxa.runtime.detect import AnswerBackendKind, EmbedBackendKind, RerankBackendKind
from noxa.runtime.manifest import resolve_artifact, resolve_gguf_artifact
from noxa.runtime.registry import RuntimeRegistry

logger = logging.getLogger(__name__)

class ModelBootstrapError(RuntimeError):
    """Raised when required models cannot be downloaded or loaded at startup."""


def configure_hf_auth(settings: Settings) -> None:
    if not settings.hf_token:
        return
    os.environ["HF_TOKEN"] = settings.hf_token
    os.environ["HUGGING_FACE_HUB_TOKEN"] = settings.hf_token


def _download_gguf(
    role: str,
    cache_dir: Path,
    token: str | None,
    settings: Settings,
) -> None:
    from huggingface_hub import hf_hub_download

    artifact = resolve_gguf_artifact(
        role, str(cache_dir), token=token, settings=settings
    )
    dest = cache_dir / artifact.filename
    if dest.exists():
        return
    log_model(
        "download start",
        artifact.filename,
        backend="llama_cpp",
        role=role,
        repo=artifact.repo_id,
    )
    t0 = time.perf_counter()
    hf_hub_download(
        repo_id=artifact.repo_id,
        filename=artifact.filename,
        local_dir=str(cache_dir),
        token=token,
    )
    log_model(
        "download done",
        artifact.filename,
        elapsed_ms(t0),
        backend="llama_cpp",
        repo=artifact.repo_id,
    )


def _download_hf_repo(model_id: str, token: str | None) -> None:
    from huggingface_hub import snapshot_download

    log_model("download start", model_id)
    t0 = time.perf_counter()
    snapshot_download(repo_id=model_id, token=token)
    log_model("download done", model_id, elapsed_ms(t0))


def _download_for_registry(registry: RuntimeRegistry) -> None:
    settings = registry.settings
    token = settings.hf_token
    cache_dir = Path(settings.model_cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    selection = registry.selection

    if selection.answer_backend == AnswerBackendKind.LLAMA_CPP:
        for role in ("answer_fast", "answer_default"):
            _download_gguf(role, cache_dir, token, settings)

    if selection.answer_backend == AnswerBackendKind.TORCH:
        for role in ("answer_fast", "answer_default"):
            _download_hf_repo(resolve_artifact(role, "torch_model", settings), token)

    if selection.embed_backend == EmbedBackendKind.ONNX:
        _download_hf_repo(registry.embed.model_id, token)
    elif selection.embed_backend == EmbedBackendKind.TORCH:
        _download_hf_repo(resolve_artifact("embed", "model", settings), token)

    if selection.rerank_backend == RerankBackendKind.ONNX:
        _download_hf_repo(registry.rerank.model_id, token)
    elif selection.rerank_backend == RerankBackendKind.TORCH:
        _download_hf_repo(resolve_artifact("rerank", "model", settings), token)


def _warm_for_registry(registry: RuntimeRegistry) -> None:
    t_warm = time.perf_counter()
    log_model("warm start", "noxa", profile=registry.selection.profile.value)

    embed = registry.embed
    t0 = time.perf_counter()
    embed.encode(["warm passage"], "passage")
    log_model(
        "warm infer",
        embed.model_id,
        elapsed_ms(t0),
        role="embedding",
        backend=embed.backend_id,
    )

    rerank = registry.rerank
    t0 = time.perf_counter()
    rerank.score("warm query", ["warm passage"])
    log_model(
        "warm infer",
        rerank.model_id,
        elapsed_ms(t0),
        role="reranker",
        backend=rerank.backend_id,
    )

    for role in ("answer_fast", "answer_default"):
        backend = registry.answer_for_role(role)
        t0 = time.perf_counter()
        backend.generate(
            "warm",
            [{"text": "warm doc"}],
            max_output_tokens=8,
        )
        log_model(
            "warm infer",
            backend.model_id,
            elapsed_ms(t0),
            role=role,
            backend=backend.backend_id,
        )

    log_model("warm done", "all", elapsed_ms(t_warm))


async def bootstrap_models(settings: Settings, registry: RuntimeRegistry) -> None:
    if not settings.preload_models:
        logger.info("Model preload disabled (NOXA_PRELOAD_MODELS=false)")
        return

    t_bootstrap = time.perf_counter()
    log_model("bootstrap start", "noxa", profile=registry.selection.profile.value)
    configure_hf_auth(settings)

    try:
        require_ml()
    except MLDependencyError as exc:
        raise ModelBootstrapError(
            "ML dependencies required for startup. Run: uv sync --extra ml-cpu"
        ) from exc

    try:
        t0 = time.perf_counter()
        await asyncio.to_thread(_download_for_registry, registry)
        log_model("bootstrap download done", "noxa", elapsed_ms(t0))

        t0 = time.perf_counter()
        await asyncio.to_thread(_warm_for_registry, registry)
        log_model("bootstrap warm done", "noxa", elapsed_ms(t0))
    except Exception as exc:
        raise ModelBootstrapError(f"Model bootstrap failed: {exc}") from exc

    log_model("bootstrap done", "noxa", elapsed_ms(t_bootstrap))
