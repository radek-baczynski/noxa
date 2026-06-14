from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Protocol

from noxa.config import AnswerSize
from noxa.llm_perf import LlmPerfStats
from noxa.schemas import Citation

EmbedKind = Literal["query", "passage"]


@dataclass
class AnswerResult:
    answer: str
    abstained: bool
    citations: list[Citation] = field(default_factory=list)
    confidence: str | None = None
    raw_response: str = ""
    model_prompt: str = ""
    generation_params: dict[str, object] | None = None
    incomplete: bool = False
    backend: str = ""
    status: str | None = None
    llm_perf: LlmPerfStats | None = None


class EmbedBackend(Protocol):
    backend_id: str
    model_id: str

    def encode(self, texts: list[str], kind: EmbedKind) -> list[list[float]]: ...


class RerankBackend(Protocol):
    backend_id: str
    model_id: str

    def score(self, query: str, passages: list[str]) -> list[float]: ...


class AnswerBackend(Protocol):
    backend_id: str
    model_id: str

    def generate(
        self,
        query: str,
        documents: list[dict[str, str]],
        max_output_tokens: int,
        answer_size: AnswerSize = AnswerSize.MEDIUM,
    ) -> AnswerResult: ...
