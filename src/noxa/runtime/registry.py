from __future__ import annotations

import logging
from dataclasses import dataclass, field

from noxa.config import Settings
from noxa.runtime.detect import (
    Capabilities,
    LlamaOffloadConfig,
    RuntimeProfile,
    llama_offload_config,
    log_inference_devices,
    probe_capabilities,
    require_llama_cpp,
    resolve_profile,
)
from noxa.runtime.interfaces import AnswerBackend, EmbedBackend, RerankBackend

logger = logging.getLogger(__name__)


@dataclass
class RuntimeSelection:
    profile: RuntimeProfile


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
        from noxa.runtime.answer_llama import LlamaAnswerBackend

        offload = llama_offload_config(self.capabilities, self.selection.profile)
        return LlamaAnswerBackend(
            role=role,
            offload=offload,
            model_cache_dir=self.settings.model_cache_dir,
            hf_token=self.settings.hf_token,
            settings=self.settings,
        )

    @classmethod
    def from_settings(cls, settings: Settings) -> RuntimeRegistry:
        caps = probe_capabilities()
        require_llama_cpp(caps)
        profile = resolve_profile(settings.runtime_profile, caps)
        selection = RuntimeSelection(profile=profile)
        offload = llama_offload_config(caps, profile)
        embed = cls._build_embed_backend(offload, settings)
        rerank = cls._build_rerank_backend(offload, settings)
        logger.info(
            "Runtime registry profile=%s backend=llama_cpp embed=%s rerank=%s",
            selection.profile.value,
            embed.model_id,
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
    def _build_embed_backend(
        offload: LlamaOffloadConfig,
        settings: Settings,
    ) -> EmbedBackend:
        from noxa.runtime.embed_llama import LlamaEmbedBackend

        return LlamaEmbedBackend(
            offload=offload,
            model_cache_dir=settings.model_cache_dir,
            hf_token=settings.hf_token,
            batch_size=settings.embedding_batch_size,
            settings=settings,
        )

    @staticmethod
    def _build_rerank_backend(
        offload: LlamaOffloadConfig,
        settings: Settings,
    ) -> RerankBackend:
        from noxa.runtime.rerank_llama import LlamaRerankBackend

        return LlamaRerankBackend(
            offload=offload,
            model_cache_dir=settings.model_cache_dir,
            hf_token=settings.hf_token,
            settings=settings,
        )
