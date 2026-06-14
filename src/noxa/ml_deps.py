from __future__ import annotations


class MLDependencyError(RuntimeError):
    def __init__(self) -> None:
        super().__init__(
            "llama-cpp-python is not installed. Run: uv sync --extra ml"
        )


def require_ml() -> None:
    from noxa.runtime.detect import probe_capabilities, require_llama_cpp

    caps = probe_capabilities()
    try:
        require_llama_cpp(caps)
    except RuntimeError as exc:
        raise MLDependencyError() from exc
