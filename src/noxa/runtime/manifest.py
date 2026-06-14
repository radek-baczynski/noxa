from __future__ import annotations

import copy
from typing import Any

from noxa.config import Settings
from noxa.runtime.artifacts import GgufArtifact
from noxa.runtime.hf_resolve import gguf_artifact_from_repo

DEFAULT_MODEL_MANIFEST: dict[str, dict[str, Any]] = {
    "answer_fast": {
        "model": "unsloth/Qwen3-0.6B-GGUF",
        "gguf_quant": "Q4_K_M",
    },
    "answer_default": {
        "model": "unsloth/Qwen3-1.7B-GGUF",
        "gguf_quant": "Q4_K_M",
    },
    "embed": {
        "model": "nomic-ai/nomic-embed-text-v1.5-GGUF",
        "gguf_quant": "Q4_K_M",
    },
    "rerank": {
        "model": "Voodisss/Qwen3-Reranker-0.6B-GGUF-llama_cpp",
        "gguf_quant": "Q4_K_M",
    },
}

MODEL_MANIFEST = DEFAULT_MODEL_MANIFEST


def effective_manifest(settings: Settings | None = None) -> dict[str, dict[str, Any]]:
    manifest = copy.deepcopy(DEFAULT_MODEL_MANIFEST)
    if settings is None:
        return manifest

    if settings.answer_model_fast:
        manifest["answer_fast"]["model"] = settings.answer_model_fast
    if settings.answer_model_default:
        manifest["answer_default"]["model"] = settings.answer_model_default
    if settings.answer_gguf_quant:
        manifest["answer_fast"]["gguf_quant"] = settings.answer_gguf_quant
        manifest["answer_default"]["gguf_quant"] = settings.answer_gguf_quant
        manifest["embed"]["gguf_quant"] = settings.answer_gguf_quant
        manifest["rerank"]["gguf_quant"] = settings.answer_gguf_quant

    if settings.embed_model:
        manifest["embed"]["model"] = settings.embed_model
    if settings.rerank_model:
        manifest["rerank"]["model"] = settings.rerank_model

    return manifest


def resolve_artifact(
    role: str,
    key: str,
    settings: Settings | None = None,
) -> Any:
    role_cfg = effective_manifest(settings).get(role, {})
    if key not in role_cfg:
        raise KeyError(f"No artifact for role={role!r} key={key!r}")
    return role_cfg[key]


def resolve_gguf_artifact(
    role: str,
    cache_dir: str,
    *,
    token: str | None = None,
    settings: Settings | None = None,
) -> GgufArtifact:
    repo_id = resolve_artifact(role, "model", settings)
    quant = resolve_artifact(role, "gguf_quant", settings)
    return gguf_artifact_from_repo(repo_id, quant=quant, token=token)
