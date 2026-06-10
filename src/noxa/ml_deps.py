from __future__ import annotations


class MLDependencyError(RuntimeError):
    def __init__(self) -> None:
        super().__init__(
            "ML dependencies not installed. Run: uv sync --extra ml-cpu "
            "(or --extra ml for torch fallback)"
        )


def require_ml() -> None:
    from noxa.runtime.detect import probe_capabilities

    caps = probe_capabilities()
    if not (caps.has_llama_cpp or caps.has_onnxruntime or caps.has_torch):
        raise MLDependencyError()


def infer_device() -> str:
    """Pick the best available PyTorch inference device."""
    from noxa.runtime.detect import infer_device as _infer

    return _infer()
