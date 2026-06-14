from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


def _tps(tokens: int, ms: float) -> float:
    if tokens <= 0 or ms <= 0:
        return 0.0
    return round(tokens / (ms / 1000.0), 1)


@dataclass
class LlmPerfStats:
    prefill_ms: float = 0.0
    prefill_tokens: int = 0
    prefill_tps: float = 0.0
    decode_ms: float = 0.0
    decode_tokens: int = 0
    decode_tps: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def perf_from_llama(llm: object, usage: dict[str, int] | None) -> LlmPerfStats:
    import llama_cpp

    perf = llama_cpp.llama_perf_context(llm.ctx)  # type: ignore[attr-defined]
    prefill_ms = float(perf.t_p_eval_ms)
    decode_ms = float(perf.t_eval_ms)
    prefill_tokens = int(perf.n_p_eval)
    decode_tokens = int(perf.n_eval)
    prompt_tokens = int((usage or {}).get("prompt_tokens", prefill_tokens))
    completion_tokens = int((usage or {}).get("completion_tokens", decode_tokens))
    return LlmPerfStats(
        prefill_ms=round(prefill_ms, 2),
        prefill_tokens=prefill_tokens,
        prefill_tps=_tps(prefill_tokens, prefill_ms),
        decode_ms=round(decode_ms, 2),
        decode_tokens=decode_tokens,
        decode_tps=_tps(decode_tokens, decode_ms),
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )
