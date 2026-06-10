from __future__ import annotations

from noxa.chunker import count_tokens
from noxa.schemas import ScoredPassage


def pack_context(
    query: str,
    passages: list[ScoredPassage],
    max_tokens: int,
) -> tuple[str, list[tuple[int, ScoredPassage]]]:
    blocks: list[str] = []
    kept: list[tuple[int, ScoredPassage]] = []
    used = count_tokens(
        "Answer the question using only the provided sources.\nQuestion:\n"
        + query
        + "\nSources:\nAnswer:"
    )
    for i, item in enumerate(passages, start=1):
        title = item.passage.title or "Untitled"
        url = item.passage.url or ""
        block = (
            f"[{i}] {title}\nURL: {url}\nPassage:\n{item.passage.text.strip()}\n"
        )
        block_tokens = count_tokens(block)
        if used + block_tokens > max_tokens:
            break
        blocks.append(block)
        kept.append((i, item))
        used += block_tokens
    formatted = "\n".join(blocks)
    return formatted, kept
