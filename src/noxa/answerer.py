from __future__ import annotations

from noxa.config import AnswerSize
from noxa.runtime.interfaces import AnswerResult
from noxa.runtime.registry import RuntimeRegistry

__all__ = ["Answerer"]


class Answerer:
    def __init__(self, registry: RuntimeRegistry, role: str) -> None:
        self.registry = registry
        self.role = role
        self.backend = registry.answer_for_role(role)

    def answer(
        self,
        query: str,
        documents: list[dict[str, str]],
        max_output_tokens: int = 2048,
        answer_size: AnswerSize = AnswerSize.MEDIUM,
    ) -> AnswerResult:
        return self.backend.generate(
            query,
            documents,
            max_output_tokens,
            answer_size=answer_size,
        )
