from __future__ import annotations

import logging
from dataclasses import dataclass, field

from noxa.config import Settings
from noxa.runtime.detect import (
    AnswerBackendKind,
    Capabilities,
    EmbedBackendKind,
    RerankBackendKind,
    RuntimeProfile,
    llama_offload_config,
    log_inference_devices,
    pick_answer_backend,
    pick_embed_backend,
    pick_rerank_backend,
    probe_capabilities,
    resolve_profile,
)
from noxa.runtime.interfaces import AnswerBackend, EmbedBackend, RerankBackend

logger = logging.getLogger(__name__)


@dataclass
class RuntimeSelection:
    profile: RuntimeProfile
    answer_backend: AnswerBackendKind
    embed_backend: EmbedBackendKind
    rerank_backend: RerankBackendKind


@dataclass
class RuntimeRegistry:
    capabilities: Capabilities
    selection: RuntimeSelection
    embed: EmbedBackend
    rerank: RerankBackend
    settings: Settings
    _answer_cache: dict[str, AnswerBackend] = field(default_factory=dict, repr=False)

    def answer_for_role(self, role: str) -> AnswerBackend:
        if role not in self._answer_cache:
            self._answer_cache[role] = self._build_answer_backend(role)
        return self._answer_cache[role]

    def _build_answer_backend(self, role: str) -> AnswerBackend:
        kind = self.selection.answer_backend
        settings = self.settings
        if kind == AnswerBackendKind.LLAMA_CPP:
            from noxa.runtime.answer_llama import LlamaAnswerBackend

            offload = llama_offload_config(self.capabilities, self.selection.profile)
            return LlamaAnswerBackend(
                role=role,
                offload=offload,
                model_cache_dir=settings.model_cache_dir,
                hf_token=settings.hf_token,
                settings=settings,
            )
        from noxa.runtime.answer_torch import TorchJsonAnswerBackend

        return TorchJsonAnswerBackend(role=role, settings=settings)

    @classmethod
    def from_settings(cls, settings: Settings) -> RuntimeRegistry:
        caps = probe_capabilities()
        profile = resolve_profile(settings.runtime_profile, caps)
        selection = RuntimeSelection(
            profile=profile,
            answer_backend=pick_answer_backend(
                settings.answer_backend, caps, profile
            ),
            embed_backend=pick_embed_backend(settings.embed_backend, caps),
            rerank_backend=pick_rerank_backend(settings.rerank_backend, caps),
        )
        embed = cls._build_embed_backend(selection.embed_backend, settings)
        rerank = cls._build_rerank_backend(selection.rerank_backend, settings)
        logger.info(
            "Runtime registry profile=%s answer=%s embed=%s/%s rerank=%s/%s",
            selection.profile.value,
            selection.answer_backend.value,
            selection.embed_backend.value,
            embed.model_id,
            selection.rerank_backend.value,
            rerank.model_id,
        )
        log_inference_devices(caps, selection, logger=logger)
        return cls(
            capabilities=caps,
            selection=selection,
            embed=embed,
            rerank=rerank,
            settings=settings,
        )

    @staticmethod
    def _build_embed_backend(kind: EmbedBackendKind, settings: Settings) -> EmbedBackend:
        if kind == EmbedBackendKind.ONNX:
            from noxa.runtime.embed_onnx import OnnxEmbedBackend

            return OnnxEmbedBackend(
                batch_size=settings.embedding_batch_size,
                settings=settings,
            )
        from noxa.runtime.embed_torch import TorchEmbedBackend

        return TorchEmbedBackend(
            batch_size=settings.embedding_batch_size,
            settings=settings,
        )

    @staticmethod
    def _build_rerank_backend(
        kind: RerankBackendKind,
        settings: Settings,
    ) -> RerankBackend:
        if kind == RerankBackendKind.ONNX:
            from noxa.runtime.rerank_onnx import OnnxRerankBackend

            return OnnxRerankBackend(settings=settings)
        from noxa.runtime.rerank_torch import TorchRerankBackend

        return TorchRerankBackend(settings=settings)
