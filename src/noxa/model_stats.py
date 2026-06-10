from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Literal

from noxa.llm_perf import LlmPerfStats

InferKind = Literal["embedding", "rerank", "answer"]


@dataclass
class ModelInferStats:
    embedding_time: int = 0
    embedding_calls: int = 0
    rerank_time: int = 0
    rerank_calls: int = 0
    answer_time: int = 0
    answer_calls: int = 0
    answer_llm: LlmPerfStats | None = None

    def record(self, kind: InferKind, time_ms: int, calls: int = 1) -> None:
        if calls <= 0 or time_ms < 0:
            return
        if kind == "embedding":
            self.embedding_time += time_ms
            self.embedding_calls += calls
        elif kind == "rerank":
            self.rerank_time += time_ms
            self.rerank_calls += calls
        elif kind == "answer":
            self.answer_time += time_ms
            self.answer_calls += calls

    def set_answer_llm(self, perf: LlmPerfStats | None) -> None:
        if perf is not None:
            self.answer_llm = perf

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "embedding_time": self.embedding_time,
            "embedding_calls": self.embedding_calls,
            "rerank_time": self.rerank_time,
            "rerank_calls": self.rerank_calls,
            "answer_time": self.answer_time,
            "answer_calls": self.answer_calls,
        }
        if self.answer_llm is not None:
            out["answer_llm"] = self.answer_llm.to_dict()
        return out


_model_infer_stats: ContextVar[ModelInferStats | None] = ContextVar(
    "model_infer_stats",
    default=None,
)


def reset_model_infer_stats() -> None:
    _model_infer_stats.set(ModelInferStats())


def get_model_infer_stats() -> ModelInferStats:
    stats = _model_infer_stats.get()
    if stats is None:
        stats = ModelInferStats()
        _model_infer_stats.set(stats)
    return stats
