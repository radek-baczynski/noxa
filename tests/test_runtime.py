from __future__ import annotations

import pytest

from noxa.answer_contract import (
    build_answer_prompt,
    parse_answer_json,
    remap_citation_source_ids,
    select_documents_for_answer,
    validate_answer_grounding,
)
from noxa.schemas import Citation
from noxa.runtime.detect import (
    Capabilities,
    LlamaOffloadConfig,
    RuntimeProfile,
    build_inference_device_plan,
    describe_llama_device,
    require_llama_cpp,
)
from noxa.runtime.hf_resolve import gguf_artifact_from_repo
from noxa.runtime.registry import RuntimeRegistry, RuntimeSelection


def _caps(**overrides: bool) -> Capabilities:
    base = dict(
        platform_system="Darwin",
        machine="arm64",
        has_llama_cpp=False,
        has_cuda=False,
        has_metal=False,
        cpu_count=8,
    )
    base.update(overrides)
    return Capabilities(**base)


def test_require_llama_cpp_raises_when_missing() -> None:
    with pytest.raises(RuntimeError, match="llama-cpp-python"):
        require_llama_cpp(_caps())


def test_resolve_answer_output_tokens_scales_by_size() -> None:
    from noxa.config import AnswerSize, resolve_answer_output_tokens

    assert resolve_answer_output_tokens(2048, AnswerSize.SHORT) == 512
    assert resolve_answer_output_tokens(2048, AnswerSize.MEDIUM) == 2048
    assert resolve_answer_output_tokens(2048, AnswerSize.COMPREHENSIVE) == 3072


def test_build_answer_prompt_includes_answer_size_hint() -> None:
    from noxa.config import AnswerSize

    short_prompt = build_answer_prompt(
        "What is asyncio?",
        [{"text": "asyncio docs"}],
        answer_size=AnswerSize.SHORT,
    )
    comprehensive_prompt = build_answer_prompt(
        "What is asyncio?",
        [{"text": "asyncio docs"}],
        answer_size=AnswerSize.COMPREHENSIVE,
    )
    assert "1-3 sentences" in short_prompt
    assert "thorough answer" in comprehensive_prompt


def test_build_answer_messages_uses_chat_roles() -> None:
    from noxa.answer_contract import build_answer_messages

    messages = build_answer_messages(
        "When was Town launched?",
        [{"text": "Town launched in 2024."}],
    )
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert "Match answer length to the question" in messages[0]["content"]
    assert "When was Town launched?" in messages[1]["content"]
    assert "[1]" in messages[1]["content"]


def test_build_answer_prompt_numbers_sources() -> None:
    prompt = build_answer_prompt(
        "What is asyncio?",
        [{"text": "asyncio docs"}, {"text": "event loop guide"}],
    )
    assert "[1]" in prompt
    assert "[2]" in prompt
    assert "asyncio docs" in prompt
    assert "SYSTEM:" in prompt
    assert "USER:" in prompt


def test_select_documents_for_answer_passthrough_preserves_order() -> None:
    documents = [
        {"text": "first", "source_id": 1},
        {"text": "second", "source_id": 2},
    ]
    selected, citation_ids = select_documents_for_answer("any query", documents)
    assert selected == documents
    assert citation_ids == [1, 2]


def test_validate_answer_grounding_rejects_too_short_answer() -> None:
    from noxa.config import AnswerSize

    documents = [{"text": "Town is an AI workspace assistant.", "source_id": 1}]
    assert not validate_answer_grounding(
        "Town.com",
        documents,
        answer_size=AnswerSize.COMPREHENSIVE,
    )


def test_validate_answer_grounding_accepts_short_factual_answer() -> None:
    from noxa.config import AnswerSize

    documents = [
        {
            "text": "Town launched in 2024 to address tax complexity for small businesses.",
            "source_id": 1,
        }
    ]
    assert validate_answer_grounding(
        "Town was launched in 2024 [1].",
        documents,
        answer_size=AnswerSize.COMPREHENSIVE,
    )


def test_parse_answer_json_strips_lfm_prose_after_json() -> None:
    raw = (
        '{"answer": "Town.com is an AI assistant.", "abstain": false, '
        '"citations": [1], "confidence": "high"}\n\n'
        "Wait, but the user said to answer using only the provided sources..."
    )
    parsed = parse_answer_json(raw)
    assert parsed["answer"] == "Town.com is an AI assistant."
    assert parsed["citations"] == [1]


def test_remap_citation_source_ids_maps_prompt_index_to_source_id() -> None:
    citations = [
        Citation(source_id=1, url=None, title=None),
        Citation(source_id=2, url=None, title=None),
    ]
    remap_citation_source_ids(citations, [1, 4])
    assert citations[0].source_id == 1
    assert citations[1].source_id == 4


def test_build_answer_prompt_has_no_query_specific_rules() -> None:
    prompt = build_answer_prompt(
        "Whats people think about poke.com?",
        [{"text": "users love it"}],
    )
    assert "opinions, reviews, or sentiment" not in prompt
    assert "geographic town" not in prompt
    assert "Whats people think about poke.com?" in prompt


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


def test_parse_answer_json_fixes_malformed_citations() -> None:
    raw = (
        '{"answer": "Poke is promising.", "abstain": false, '
        '"citations": [1], [2], [3], "confidence": "high"}'
    )
    parsed = parse_answer_json(raw)
    assert parsed["answer"] == "Poke is promising."
    assert parsed["abstain"] is False
    assert parsed["citations"] == [1, 2, 3]
    assert parsed["confidence"] == "high"


def test_parse_answer_json_strips_thinking_block() -> None:
    raw = (
        '{"answer": "yes", "abstain": false, "citations": [1], "confidence": "high"}\n\n'
        "<think>\nLong reasoning here..."
    )
    parsed = parse_answer_json(raw)
    assert parsed["answer"] == "yes"
    assert parsed["abstain"] is False


def test_parse_answer_json_salvages_poke_dump_shape() -> None:
    raw = (
        ' {"answer": "Poke.com is a proactive AI assistant.", "abstain": false, '
        '"citations": [1], [2], [3], [4], [5], [6], [7], "confidence": "high"}\n\n'
        "<think>\nThinking Process:\n1. Analyze..."
    )
    parsed = parse_answer_json(raw)
    assert "Poke.com" in parsed["answer"]
    assert parsed["abstain"] is False
    assert parsed["citations"] == [1, 2, 3, 4, 5, 6, 7]


@pytest.mark.integration
def test_gguf_artifact_from_repo() -> None:
    artifact = gguf_artifact_from_repo("unsloth/Qwen3-0.6B-GGUF")
    assert artifact.repo_id == "unsloth/Qwen3-0.6B-GGUF"
    assert artifact.filename.endswith(".gguf")
    assert "Q4_K_M" in artifact.filename.upper()


def test_effective_manifest_env_overrides(monkeypatch) -> None:
    monkeypatch.setenv("NOXA_ANSWER_MODEL_FAST", "unsloth/Qwen3-0.6B-GGUF")
    monkeypatch.setenv("NOXA_EMBED_MODEL", "nomic-ai/nomic-embed-text-v1.5-GGUF")
    monkeypatch.setenv("NOXA_RERANK_MODEL", "Voodisss/Qwen3-Reranker-0.6B-GGUF-llama_cpp")
    from noxa.config import Settings
    from noxa.runtime.manifest import effective_manifest, resolve_artifact

    settings = Settings()
    manifest = effective_manifest(settings)
    assert manifest["answer_fast"]["model"] == "unsloth/Qwen3-0.6B-GGUF"
    assert resolve_artifact("embed", "model", settings) == (
        "nomic-ai/nomic-embed-text-v1.5-GGUF"
    )


def test_describe_llama_device_metal_on_mac() -> None:
    caps = _caps(has_llama_cpp=True, has_metal=True)
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
    caps = _caps(has_llama_cpp=True, has_metal=True)
    selection = RuntimeSelection(profile=RuntimeProfile.MAC_LOCAL)
    plan = build_inference_device_plan(caps, selection)
    assert plan.answer_device == "metal (8 threads, all layers)"
    assert plan.embed_device == "metal (8 threads, all layers)"
    assert plan.rerank_device == "metal (8 threads, all layers)"


def test_registry_from_settings_without_ml(monkeypatch) -> None:
    monkeypatch.setattr(
        "noxa.runtime.registry.probe_capabilities",
        lambda: _caps(has_llama_cpp=True, has_metal=True),
    )
    from noxa.config import get_settings

    registry = RuntimeRegistry.from_settings(get_settings())
    assert registry.embed.backend_id == "llama_cpp"
    assert registry.embed.model_id
    assert registry.rerank.backend_id == "llama_cpp"
    assert registry.rerank.model_id
