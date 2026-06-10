from __future__ import annotations

import logging
import re
import time

import tiktoken

from noxa.request_context import log_prefix
from noxa.schemas import ExtractedPage, Passage
from noxa.text_clean import clean_page_markdown

logger = logging.getLogger(__name__)

_ENC: tiktoken.Encoding | None = None


def _encoder() -> tiktoken.Encoding:
    global _ENC
    if _ENC is None:
        t0 = time.perf_counter()
        try:
            _ENC = tiktoken.get_encoding("cl100k_base")
            enc_name = "cl100k_base"
        except Exception:
            _ENC = tiktoken.get_encoding("gpt2")
            enc_name = "gpt2"
        logger.info(
            "%schunk encoder loaded encoding=%s latency=%dms",
            log_prefix(),
            enc_name,
            int((time.perf_counter() - t0) * 1000),
        )
    return _ENC


def count_tokens(text: str) -> int:
    return len(_encoder().encode(text))


def _split_oversized_text(
    page: ExtractedPage,
    text: str,
    start_char: int,
    idx: int,
    passage_tokens: int,
    overlap_tokens: int,
) -> tuple[list[Passage], int]:
    """Split text that exceeds passage_tokens using a sliding token window."""
    enc = _encoder()
    tokens = enc.encode(text)
    step = max(1, passage_tokens - overlap_tokens)
    passages: list[Passage] = []
    start = 0
    while start < len(tokens):
        end = min(start + passage_tokens, len(tokens))
        chunk_tokens = tokens[start:end]
        chunk_text = enc.decode(chunk_tokens)
        passages.append(
            Passage(
                passage_id=f"{page.source_id}#{idx}",
                source_id=page.source_id,
                url=page.url,
                title=page.title,
                text=chunk_text.strip(),
                start_char=start_char,
                end_char=start_char + len(chunk_text),
                token_count=len(chunk_tokens),
                source_rank=page.source_rank,
            )
        )
        idx += 1
        if end >= len(tokens):
            break
        start += step
    return passages, idx


def chunk_pages(
    pages: list[ExtractedPage],
    passage_tokens: int,
    overlap_tokens: int,
) -> list[Passage]:
    passages: list[Passage] = []
    for page_idx, page in enumerate(pages):
        page_t0 = time.perf_counter()
        text_len = len(page.text)
        cleaned_text = clean_page_markdown(page.text)
        if not cleaned_text:
            logger.info(
                "%schunk page skip empty url=%s",
                log_prefix(),
                page.url,
            )
            continue

        page_tokens = count_tokens(cleaned_text)
        logger.info(
            "%schunk page start [%d/%d] url=%s chars=%d tokens=%d",
            log_prefix(),
            page_idx + 1,
            len(pages),
            page.url,
            text_len,
            page_tokens,
        )

        split_t0 = time.perf_counter()
        blocks = re.split(r"\n\s*\n", cleaned_text)
        non_empty_blocks = sum(1 for b in blocks if b.strip())
        logger.info(
            "%schunk page split blocks=%d (non_empty=%d) latency=%dms",
            log_prefix(),
            len(blocks),
            non_empty_blocks,
            int((time.perf_counter() - split_t0) * 1000),
        )

        page_passages_before = len(passages)
        current = ""
        start_char = 0
        block_start = 0
        idx = 0
        for block_idx, block in enumerate(blocks):
            block = block.strip()
            if not block:
                block_start += 2
                continue

            if current:
                candidate = f"{current}\n\n{block}".strip()
                if count_tokens(candidate) <= passage_tokens:
                    current = candidate
                    block_start += len(block) + 2
                    continue
                passages.append(_make_passage(page, current, start_char, idx))
                idx += 1
                current = ""

            start_char = block_start
            block_tokens = count_tokens(block)
            if block_tokens <= passage_tokens:
                current = block
            else:
                split, idx = _split_oversized_text(
                    page, block, start_char, idx, passage_tokens, overlap_tokens
                )
                passages.extend(split)
                logger.info(
                    "%schunk oversized block=%d/%d tokens=%d split_into=%d",
                    log_prefix(),
                    block_idx + 1,
                    len(blocks),
                    block_tokens,
                    len(split),
                )
                current = ""

            block_start += len(block) + 2

        if current.strip():
            passages.append(_make_passage(page, current, start_char, idx))

        page_passages = len(passages) - page_passages_before
        logger.info(
            "%schunk page done url=%s passages=%d latency=%dms",
            log_prefix(),
            page.url,
            page_passages,
            int((time.perf_counter() - page_t0) * 1000),
        )

    logger.info(
        "%schunk all pages done total_passages=%d",
        log_prefix(),
        len(passages),
    )
    return passages


def _make_passage(page: ExtractedPage, text: str, start_char: int, idx: int) -> Passage:
    end_char = start_char + len(text)
    pid = f"{page.source_id}#{idx}"
    stripped = text.strip()
    return Passage(
        passage_id=pid,
        source_id=page.source_id,
        url=page.url,
        title=page.title,
        text=stripped,
        start_char=start_char,
        end_char=end_char,
        token_count=count_tokens(stripped),
        source_rank=page.source_rank,
    )


def _trim_to_tokens(text: str, max_tokens: int) -> str:
    enc = _encoder()
    tokens = enc.encode(text)
    return enc.decode(tokens[:max_tokens])


def _tail_tokens(text: str, overlap_tokens: int) -> str:
    if overlap_tokens <= 0:
        return ""
    enc = _encoder()
    tokens = enc.encode(text)
    return enc.decode(tokens[-overlap_tokens:])
