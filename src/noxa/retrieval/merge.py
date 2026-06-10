from __future__ import annotations

from noxa.config import Settings
from noxa.schemas import ScoredPassage


def merge_candidates(
    bm25: list[ScoredPassage],
    embedding: list[ScoredPassage],
    top_k: int,
    settings: Settings,
) -> list[ScoredPassage]:
    merged: dict[str, ScoredPassage] = {}

    for rank, item in enumerate(bm25, start=1):
        pid = item.passage.passage_id
        score = settings.bm25_weight * (1.0 / rank)
        if pid in merged:
            merged[pid].merged_score = (merged[pid].merged_score or 0) + score
            merged[pid].bm25_score = item.bm25_score
        else:
            merged[pid] = ScoredPassage(
                passage=item.passage,
                bm25_score=item.bm25_score,
                merged_score=score,
            )

    for rank, item in enumerate(embedding, start=1):
        pid = item.passage.passage_id
        score = settings.embedding_weight * (1.0 / rank)
        if pid in merged:
            merged[pid].merged_score = (merged[pid].merged_score or 0) + score
            merged[pid].embedding_score = item.embedding_score
        else:
            merged[pid] = ScoredPassage(
                passage=item.passage,
                embedding_score=item.embedding_score,
                merged_score=score,
            )

    out = sorted(merged.values(), key=lambda s: s.merged_score or 0.0, reverse=True)
    return out[:top_k]
