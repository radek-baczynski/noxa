from __future__ import annotations

import logging
import os
import platform
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from noxa.runtime.registry import RuntimeSelection


class RuntimeProfile(StrEnum):
    AUTO = "auto"
    MAC_LOCAL = "mac-local"
    CLOUD_CPU = "cloud-cpu"
    CLOUD_GPU = "cloud-gpu"


class AnswerBackendKind(StrEnum):
    LLAMA_CPP = "llama_cpp"
    TORCH = "torch"


class EmbedBackendKind(StrEnum):
    ONNX = "onnx"
    TORCH = "torch"


class RerankBackendKind(StrEnum):
    ONNX = "onnx"
    TORCH = "torch"


@dataclass
class Capabilities:
    platform_system: str
    machine: str
    has_llama_cpp: bool
    has_onnxruntime: bool
    has_torch: bool
    has_cuda: bool
    has_mps: bool
    cpu_count: int


@dataclass
class LlamaOffloadConfig:
    n_gpu_layers: int
    n_threads: int
    n_ctx: int = 4096


def probe_capabilities() -> Capabilities:
    has_llama_cpp = _can_import("llama_cpp")
    has_onnxruntime = _can_import("onnxruntime")
    has_torch = _can_import("torch")
    has_cuda = False
    has_mps = False
    if has_torch:
        import torch

        has_cuda = bool(torch.cuda.is_available())
        has_mps = bool(getattr(torch.backends, "mps", None) and torch.mps.is_available())
    return Capabilities(
        platform_system=platform.system(),
        machine=platform.machine().lower(),
        has_llama_cpp=has_llama_cpp,
        has_onnxruntime=has_onnxruntime,
        has_torch=has_torch,
        has_cuda=has_cuda,
        has_mps=has_mps,
        cpu_count=os.cpu_count() or 4,
    )


def infer_device() -> str:
    """Pick the best available PyTorch inference device."""
    if not _can_import("torch"):
        return "cpu"
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def resolve_profile(profile: str, caps: Capabilities) -> RuntimeProfile:
    if profile != RuntimeProfile.AUTO:
        return RuntimeProfile(profile)
    if caps.platform_system == "Darwin" and caps.machine in {"arm64", "aarch64"}:
        return RuntimeProfile.MAC_LOCAL
    if caps.has_cuda:
        return RuntimeProfile.CLOUD_GPU
    return RuntimeProfile.CLOUD_CPU


def llama_offload_config(caps: Capabilities, profile: RuntimeProfile) -> LlamaOffloadConfig:
    threads = max(4, min(caps.cpu_count, 16))
    if profile == RuntimeProfile.MAC_LOCAL or (
        caps.platform_system == "Darwin" and caps.machine in {"arm64", "aarch64"}
    ):
        return LlamaOffloadConfig(n_gpu_layers=-1, n_threads=threads)
    if caps.has_cuda or profile == RuntimeProfile.CLOUD_GPU:
        return LlamaOffloadConfig(n_gpu_layers=-1, n_threads=threads)
    return LlamaOffloadConfig(n_gpu_layers=0, n_threads=threads)


def pick_answer_backend(
    requested: str,
    caps: Capabilities,
    profile: RuntimeProfile,
) -> AnswerBackendKind:
    if requested and requested != "auto":
        return AnswerBackendKind(requested)
    if caps.has_llama_cpp:
        return AnswerBackendKind.LLAMA_CPP
    if caps.has_torch:
        return AnswerBackendKind.TORCH
    raise RuntimeError("No answer backend available (install llama-cpp-python or torch)")


def pick_embed_backend(requested: str, caps: Capabilities) -> EmbedBackendKind:
    if requested and requested != "auto":
        return EmbedBackendKind(requested)
    if caps.has_onnxruntime:
        return EmbedBackendKind.ONNX
    if caps.has_torch:
        return EmbedBackendKind.TORCH
    raise RuntimeError("No embed backend available")


def pick_rerank_backend(requested: str, caps: Capabilities) -> RerankBackendKind:
    if requested and requested != "auto":
        return RerankBackendKind(requested)
    if caps.has_onnxruntime:
        return RerankBackendKind.ONNX
    if caps.has_torch:
        return RerankBackendKind.TORCH
    raise RuntimeError("No rerank backend available")


def _can_import(module: str) -> bool:
    try:
        __import__(module)
        return True
    except ImportError:
        return False


_TORCH_DEVICE_LABELS = {
    "mps": "mps (Apple GPU)",
    "cuda": "cuda (NVIDIA GPU)",
    "cpu": "cpu",
}


def describe_torch_device(caps: Capabilities) -> str:
    if not caps.has_torch:
        return "cpu"
    return _TORCH_DEVICE_LABELS.get(infer_device(), "cpu")


def describe_onnx_device() -> str:
    """fastembed/onnxruntime runs on CPU in Noxa today."""
    if not _can_import("onnxruntime"):
        return "cpu"
    return "cpu (onnxruntime)"


def describe_llama_device(
    caps: Capabilities,
    profile: RuntimeProfile,
    offload: LlamaOffloadConfig,
) -> str:
    if offload.n_gpu_layers == 0:
        return f"cpu ({offload.n_threads} threads)"
    if (
        caps.platform_system == "Darwin"
        and caps.machine in {"arm64", "aarch64"}
    ):
        return f"metal ({offload.n_threads} threads, all layers)"
    if caps.has_cuda or profile == RuntimeProfile.CLOUD_GPU:
        return f"cuda ({offload.n_threads} threads, all layers)"
    if offload.n_gpu_layers == -1:
        return f"gpu ({offload.n_threads} threads, all layers)"
    return f"gpu ({offload.n_gpu_layers} layers, {offload.n_threads} threads)"


@dataclass
class InferenceDevicePlan:
    profile: RuntimeProfile
    answer_backend: AnswerBackendKind
    answer_device: str
    embed_backend: EmbedBackendKind
    embed_device: str
    rerank_backend: RerankBackendKind
    rerank_device: str


def build_inference_device_plan(
    caps: Capabilities,
    selection: RuntimeSelection,
) -> InferenceDevicePlan:
    if selection.answer_backend == AnswerBackendKind.TORCH:
        answer_device = describe_torch_device(caps)
    else:
        offload = llama_offload_config(caps, selection.profile)
        answer_device = describe_llama_device(caps, selection.profile, offload)

    if selection.embed_backend == EmbedBackendKind.TORCH:
        embed_device = describe_torch_device(caps)
    else:
        embed_device = describe_onnx_device()

    if selection.rerank_backend == RerankBackendKind.TORCH:
        rerank_device = describe_torch_device(caps)
    else:
        rerank_device = describe_onnx_device()

    return InferenceDevicePlan(
        profile=selection.profile,
        answer_backend=selection.answer_backend,
        answer_device=answer_device,
        embed_backend=selection.embed_backend,
        embed_device=embed_device,
        rerank_backend=selection.rerank_backend,
        rerank_device=rerank_device,
    )


def log_inference_devices(
    caps: Capabilities,
    selection: RuntimeSelection,
    *,
    logger: logging.Logger | None = None,
) -> InferenceDevicePlan:
    plan = build_inference_device_plan(caps, selection)
    log = logger or logging.getLogger("noxa.runtime")
    log.info("Inference devices profile=%s", plan.profile.value)
    log.info(
        "  answer: %s (%s)",
        plan.answer_device,
        plan.answer_backend.value,
    )
    log.info(
        "  embed:  %s (%s)",
        plan.embed_device,
        plan.embed_backend.value,
    )
    log.info(
        "  rerank: %s (%s)",
        plan.rerank_device,
        plan.rerank_backend.value,
    )
    return plan
