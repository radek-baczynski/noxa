from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def _list_embed_models() -> list[dict]:
    from fastembed import TextEmbedding

    return list(TextEmbedding.list_supported_models())


def _list_rerank_models() -> list[dict]:
    from fastembed.rerank.cross_encoder import TextCrossEncoder

    return list(TextCrossEncoder.list_supported_models())


def resolve_fastembed_embed(
    *,
    prefer_multilingual: bool = True,
    max_dim: int = 512,
) -> str:
    """Pick an embed model from fastembed's supported list (no hardcoded guesses)."""
    models = _list_embed_models()
    scored: list[tuple[int, str]] = []
    for m in models:
        model_id = m["model"]
        dim = m.get("dim") or m.get("embedding_dim") or 9999
        if dim > max_dim:
            continue
        desc = (m.get("description") or "").lower()
        mid = model_id.lower()
        score = 0
        if prefer_multilingual and (
            "multilingual" in mid or "multilingual" in desc or "multi-lingual" in desc
        ):
            score += 10
        if "e5" in mid:
            score += 8
        if "bge" in mid:
            score += 6
        if dim == 384:
            score += 4
        scored.append((score, model_id))

    if not scored:
        raise RuntimeError("No fastembed embed model matched constraints")
    scored.sort(reverse=True)
    chosen = scored[0][1]
    logger.info("Resolved fastembed embed -> %s", chosen)
    return chosen


def resolve_fastembed_rerank(
    *,
    prefer_multilingual: bool = True,
) -> str:
    """Pick a cross-encoder reranker from fastembed's supported list."""
    models = _list_rerank_models()
    scored: list[tuple[int, str]] = []
    for m in models:
        model_id = m["model"]
        mid = model_id.lower()
        desc = (m.get("description") or "").lower()
        score = 0
        if "bge-reranker" in mid:
            score += 10
        if "reranker" in mid:
            score += 6
        if prefer_multilingual and ("multilingual" in mid or "multilingual" in desc):
            score += 8
        if "v2" in mid:
            score += 4
        scored.append((score, model_id))

    if not scored:
        raise RuntimeError("No fastembed rerank model matched constraints")
    scored.sort(reverse=True)
    chosen = scored[0][1]
    logger.info("Resolved fastembed rerank -> %s", chosen)
    return chosen
