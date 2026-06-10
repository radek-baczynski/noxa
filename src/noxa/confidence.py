from __future__ import annotations

CONFIDENCE_HIGH_THRESHOLD = 0.85
CONFIDENCE_LOW_THRESHOLD = 0.60


def compute_confidence(
    *,
    abstained: bool,
    incomplete: bool,
    top_rerank_score: float | None,
    has_citations: bool,
    model_confidence: str | None = None,
) -> str:
    if abstained or incomplete:
        return "low"
    if model_confidence == "low":
        return "low"
    if top_rerank_score is None:
        return model_confidence or "low"
    if top_rerank_score >= CONFIDENCE_HIGH_THRESHOLD and has_citations:
        if model_confidence == "high":
            return "high"
        if model_confidence == "medium":
            return "medium"
        return "high"
    if top_rerank_score < CONFIDENCE_LOW_THRESHOLD:
        return "low"
    return model_confidence or "medium"
