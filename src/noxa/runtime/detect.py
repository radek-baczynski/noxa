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


@dataclass
class Capabilities:
    platform_system: str
    machine: str
    has_llama_cpp: bool
    has_cuda: bool
    has_metal: bool
    cpu_count: int


@dataclass
class LlamaOffloadConfig:
    n_gpu_layers: int
    n_threads: int
    n_ctx: int = 4096


def probe_capabilities() -> Capabilities:
    has_llama_cpp = _can_import("llama_cpp")
    has_cuda = False
    has_metal = False
    if has_llama_cpp:
        import llama_cpp

        info = llama_cpp.llama_print_system_info().decode("utf-8", errors="replace")
        has_cuda = "CUDA" in info
        has_metal = "Metal" in info or "MTL" in info
    return Capabilities(
        platform_system=platform.system(),
        machine=platform.machine().lower(),
        has_llama_cpp=has_llama_cpp,
        has_cuda=has_cuda,
        has_metal=has_metal,
        cpu_count=os.cpu_count() or 4,
    )


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


def require_llama_cpp(caps: Capabilities) -> None:
    if not caps.has_llama_cpp:
        raise RuntimeError(
            "llama-cpp-python is required. Run: uv sync --extra ml"
        )


def _can_import(module: str) -> bool:
    try:
        __import__(module)
        return True
    except ImportError:
        return False


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
    answer_device: str
    embed_device: str
    rerank_device: str


def build_inference_device_plan(
    caps: Capabilities,
    selection: RuntimeSelection,
) -> InferenceDevicePlan:
    offload = llama_offload_config(caps, selection.profile)
    device = describe_llama_device(caps, selection.profile, offload)
    return InferenceDevicePlan(
        profile=selection.profile,
        answer_device=device,
        embed_device=device,
        rerank_device=device,
    )


def log_inference_devices(
    caps: Capabilities,
    selection: RuntimeSelection,
    *,
    logger: logging.Logger | None = None,
) -> InferenceDevicePlan:
    plan = build_inference_device_plan(caps, selection)
    log = logger or logging.getLogger("noxa.runtime")
    log.info("Inference devices profile=%s backend=llama_cpp", plan.profile.value)
    log.info("  answer: %s", plan.answer_device)
    log.info("  embed:  %s", plan.embed_device)
    log.info("  rerank: %s", plan.rerank_device)
    return plan
