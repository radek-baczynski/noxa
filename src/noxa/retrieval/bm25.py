from __future__ import annotations

import re

from rank_bm25 import BM25Okapi

from noxa.config import Settings
from noxa.schemas import Passage, ScoredPassage

_WORD_RE = re.compile(r"\w+")


def _tokenize(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


def _query_words(query: str) -> list[str]:
    words = _tokenize(query)
    return [w for w in words if len(w) > 2]


def bm25_search(
    query: str,
    passages: list[Passage],
    top_k: int,
    settings: Settings,
) -> list[ScoredPassage]:
    if not passages:
        return []
    corpus = []
    for p in passages:
        title_part = (p.title or "") + " "
        url_part = (p.url or "") + " "
        corpus.append(_tokenize(title_part + url_part + p.text))
    bm25 = BM25Okapi(corpus)
    scores = bm25.get_scores(_tokenize(query))
    query_words = _query_words(query)
    scored: list[ScoredPassage] = []
    for passage, score in zip(passages, scores, strict=True):
        boosted = float(score)
        if passage.title and query_words:
            title_tokens = set(_tokenize(passage.title))
            if any(w in title_tokens for w in query_words):
                boosted += settings.bm25_title_boost
        if passage.url and query_words:
            url_tokens = set(_tokenize(passage.url))
            if any(w in url_tokens for w in query_words):
                boosted += settings.bm25_url_boost
        scored.append(ScoredPassage(passage=passage, bm25_score=boosted))
    scored.sort(key=lambda s: s.bm25_score or 0.0, reverse=True)
    return scored[:top_k]
