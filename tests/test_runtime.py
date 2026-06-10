from __future__ import annotations

import pytest

from noxa.answer_contract import build_answer_prompt, parse_answer_json
from noxa.runtime.detect import (
    AnswerBackendKind,
    Capabilities,
    EmbedBackendKind,
    LlamaOffloadConfig,
    RerankBackendKind,
    RuntimeProfile,
    build_inference_device_plan,
    describe_llama_device,
    pick_answer_backend,
    pick_embed_backend,
    pick_rerank_backend,
)
from noxa.runtime.hf_resolve import gguf_artifact_from_repo
from noxa.runtime.registry import RuntimeRegistry, RuntimeSelection


def _caps(**overrides: bool) -> Capabilities:
    base = dict(
        platform_system="Darwin",
        machine="arm64",
        has_llama_cpp=False,
        has_onnxruntime=False,
        has_torch=False,
        has_cuda=False,
        has_mps=False,
        cpu_count=8,
    )
    base.update(overrides)
    return Capabilities(**base)


def test_pick_answer_backend_priority() -> None:
    caps = _caps(has_llama_cpp=True, has_torch=True)
    assert pick_answer_backend("auto", caps, RuntimeProfile.MAC_LOCAL) == (
        AnswerBackendKind.LLAMA_CPP
    )
    caps = _caps(has_torch=True)
    assert pick_answer_backend("auto", caps, RuntimeProfile.CLOUD_CPU) == (
        AnswerBackendKind.TORCH
    )


def test_pick_embed_and_rerank_backends() -> None:
    caps = _caps(has_onnxruntime=True, has_torch=True)
    assert pick_embed_backend("auto", caps) == EmbedBackendKind.ONNX
    assert pick_rerank_backend("auto", caps) == RerankBackendKind.ONNX
    caps = _caps(has_torch=True)
    assert pick_embed_backend("auto", caps) == EmbedBackendKind.TORCH
    assert pick_rerank_backend("auto", caps) == RerankBackendKind.TORCH


def test_build_answer_prompt_numbers_sources() -> None:
    prompt = build_answer_prompt(
        "What is asyncio?",
        [{"text": "asyncio docs"}, {"text": "event loop guide"}],
    )
    assert "[1]" in prompt
    assert "[2]" in prompt
    assert "asyncio docs" in prompt
    assert "JSON" in prompt


def test_parse_answer_json_salvages_fenced_block() -> None:
    raw = 'Here is the answer:\n```json\n{"answer": "yes [1]", "abstain": false, "citations": [1], "confidence": "high"}\n```'
    parsed = parse_answer_json(raw)
    assert parsed["answer"] == "yes [1]"
    assert parsed["abstain"] is False
    assert parsed["citations"] == [1]
    assert parsed["confidence"] == "high"


def test_parse_answer_json_abstains_on_empty() -> None:
    parsed = parse_answer_json('{"answer": "", "abstain": false}')
    assert parsed["abstain"] is True


@pytest.mark.integration
def test_gguf_artifact_from_repo() -> None:
    artifact = gguf_artifact_from_repo("unsloth/Qwen3-0.6B-GGUF")
    assert artifact.repo_id == "unsloth/Qwen3-0.6B-GGUF"
    assert artifact.filename.endswith(".gguf")
    assert "Q4_K_M" in artifact.filename.upper()


def test_effective_manifest_env_overrides(monkeypatch) -> None:
    monkeypatch.setenv("NOXA_ANSWER_MODEL_FAST", "unsloth/Qwen3-0.6B-GGUF")
    monkeypatch.setenv("NOXA_EMBED_MODEL", "BAAI/bge-small-en-v1.5")
    monkeypatch.setenv("NOXA_RERANK_MODEL", "BAAI/bge-reranker-base")
    from noxa.config import Settings
    from noxa.runtime.manifest import effective_manifest, resolve_artifact

    settings = Settings()
    manifest = effective_manifest(settings)
    assert manifest["answer_fast"]["model"] == "unsloth/Qwen3-0.6B-GGUF"
    assert resolve_artifact("answer_fast", "torch_model", settings) == "Qwen/Qwen3-0.6B"
    assert resolve_artifact("embed", "model", settings) == "BAAI/bge-small-en-v1.5"


def test_effective_manifest_torch_model_from_non_gguf_override(monkeypatch) -> None:
    monkeypatch.setenv("NOXA_ANSWER_MODEL_FAST", "Qwen/Qwen3-4B")
    from noxa.config import Settings
    from noxa.runtime.manifest import resolve_artifact

    settings = Settings()
    assert resolve_artifact("answer_fast", "model", settings) == "Qwen/Qwen3-4B"
    assert resolve_artifact("answer_fast", "torch_model", settings) == "Qwen/Qwen3-4B"


def test_describe_llama_device_metal_on_mac() -> None:
    caps = _caps(has_llama_cpp=True, has_mps=True)
    offload = LlamaOffloadConfig(n_gpu_layers=-1, n_threads=8)
    assert describe_llama_device(caps, RuntimeProfile.MAC_LOCAL, offload) == (
        "metal (8 threads, all layers)"
    )


def test_describe_llama_device_cpu_on_cloud_cpu() -> None:
    caps = _caps(platform_system="Linux", machine="x86_64")
    offload = LlamaOffloadConfig(n_gpu_layers=0, n_threads=12)
    assert describe_llama_device(caps, RuntimeProfile.CLOUD_CPU, offload) == (
        "cpu (12 threads)"
    )


def test_build_inference_device_plan() -> None:
    caps = _caps(has_llama_cpp=True, has_onnxruntime=True, has_torch=True, has_mps=True)
    selection = RuntimeSelection(
        profile=RuntimeProfile.MAC_LOCAL,
        answer_backend=AnswerBackendKind.LLAMA_CPP,
        embed_backend=EmbedBackendKind.ONNX,
        rerank_backend=RerankBackendKind.ONNX,
    )
    plan = build_inference_device_plan(caps, selection)
    assert plan.answer_device == "metal (8 threads, all layers)"
    assert plan.embed_device == "cpu (onnxruntime)"
    assert plan.rerank_device == "cpu (onnxruntime)"


def test_registry_from_settings_without_ml(monkeypatch) -> None:
    monkeypatch.setattr(
        "noxa.runtime.registry.probe_capabilities",
        lambda: _caps(has_onnxruntime=True, has_llama_cpp=True),
    )
    from noxa.config import get_settings

    registry = RuntimeRegistry.from_settings(get_settings())
    assert registry.selection.answer_backend == AnswerBackendKind.LLAMA_CPP
    assert registry.selection.embed_backend == EmbedBackendKind.ONNX
    assert registry.embed.backend_id == "onnx"
    assert registry.embed.model_id
    assert registry.rerank.backend_id == "onnx"
    assert registry.rerank.model_id
