from __future__ import annotations

import math

from noxa.config import Settings
from noxa.model_timing import timed_model_op
from noxa.runtime.detect import LlamaOffloadConfig
from noxa.runtime.llama_common import load_llama, offload_key
from noxa.runtime.manifest import resolve_gguf_artifact

INSTRUCTION = (
    "Given a web search query, retrieve relevant passages that answer the query"
)
RERANK_PASSAGE_MAX_CHARS = 512
RERANK_TITLE_MAX_CHARS = 120
RERANK_URL_MAX_CHARS = 200
# Safety cap for fully formatted documents (title + url + passage).
RERANK_DOC_MAX_CHARS = (
    RERANK_TITLE_MAX_CHARS + RERANK_URL_MAX_CHARS + RERANK_PASSAGE_MAX_CHARS + 32
)


def truncate_rerank_passage(
    text: str,
    max_chars: int = RERANK_PASSAGE_MAX_CHARS,
) -> str:
    return _truncate_passage(text, max_chars)


def format_rerank_document(
    text: str,
    *,
    title: str | None = None,
    url: str | None = None,
) -> str:
    """Document string the reranker scores: title/url header + truncated passage.

    Title and URL let the model use page-level context (e.g. which site or
    entity a passage belongs to) when the passage text alone is ambiguous.
    """
    parts: list[str] = []
    if title:
        parts.append(f"Title: {_truncate_passage(title, RERANK_TITLE_MAX_CHARS)}")
    if url:
        parts.append(f"URL: {_truncate_passage(url, RERANK_URL_MAX_CHARS)}")
    parts.append(_truncate_passage(text))
    return "\n".join(parts)


def format_rerank_prompt(
    query: str,
    passage: str,
    *,
    title: str | None = None,
    url: str | None = None,
) -> str:
    return _format_pair(query, format_rerank_document(passage, title=title, url=url))


def _truncate_passage(text: str, max_chars: int = RERANK_PASSAGE_MAX_CHARS) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "..."


def _prompt_parts(query: str) -> tuple[str, str]:
    """Shared prefix (system + instruct + query) and suffix around the document."""
    prefix = (
        "<|im_start|>system\n"
        "Judge whether the Document meets the requirements based on the Query and "
        'the Instruct provided. Note that the answer can only be "yes" or "no".\n'
        "<|im_start|>user\n"
        f"<Instruct>: {INSTRUCTION}\n"
        f"<Query>: {query}\n"
        "<Document>: "
    )
    suffix = "<|im_start|>assistant\n\n\n\n\n"
    return prefix, suffix


def _format_pair(query: str, passage: str) -> str:
    prefix, suffix = _prompt_parts(query)
    return f"{prefix}{passage}{suffix}"


def _yes_no_probability(llm: object, yes_id: int, no_id: int) -> float:
    # Last-token logits from the C buffer; only the two ids we need are read,
    # avoiding the full-vocab copy that eval_logits/logits_all would do.
    logits = llm._ctx.get_logits()
    yes_logit = logits[yes_id]
    no_logit = logits[no_id]
    max_logit = max(yes_logit, no_logit)
    exp_yes = math.exp(yes_logit - max_logit)
    exp_no = math.exp(no_logit - max_logit)
    return exp_yes / (exp_yes + exp_no)


class LlamaRerankBackend:
    backend_id = "llama_cpp"

    def __init__(
        self,
        offload: LlamaOffloadConfig,
        model_cache_dir: str,
        hf_token: str | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.offload = offload
        self.model_cache_dir = model_cache_dir
        self.hf_token = hf_token
        self.settings = settings
        self.artifact = resolve_gguf_artifact(
            "rerank", model_cache_dir, token=hf_token, settings=settings
        )
        self.model_id = self.artifact.filename

    def _llm(self) -> object:
        llm, _ = load_llama(
            self.artifact.repo_id,
            self.artifact.filename,
            offload_key(self.offload),
            self.model_cache_dir,
            self.hf_token,
            "rerank",
        )
        return llm

    def score(self, query: str, passages: list[str]) -> list[float]:
        """Score formatted document strings (see format_rerank_document)."""
        if not passages:
            return []
        llm = self._llm()
        truncated = [_truncate_passage(p, RERANK_DOC_MAX_CHARS) for p in passages]
        prefix, suffix = _prompt_parts(query)
        prefix_tokens = llm.tokenize(prefix.encode("utf-8"))
        suffix_tokens = llm.tokenize(suffix.encode("utf-8"), add_bos=False)
        yes_id = llm.tokenize(b"yes", add_bos=False, special=False)[0]
        no_id = llm.tokenize(b"no", add_bos=False, special=False)[0]
        with timed_model_op(
            self.model_id,
            "infer rerank",
            infer_kind="rerank",
            infer_calls=len(passages),
            backend=self.backend_id,
            batch=len(passages),
        ):
            # Prefill the shared prefix once; per candidate, roll the KV cache
            # back to the prefix and evaluate only document + suffix tokens
            # (Llama.eval prunes cache entries past n_tokens).
            llm.reset()
            llm.eval(prefix_tokens)
            prefix_len = llm.n_tokens
            scores: list[float] = []
            for passage in truncated:
                doc_tokens = llm.tokenize(passage.encode("utf-8"), add_bos=False)
                llm.n_tokens = prefix_len
                llm.eval(doc_tokens + suffix_tokens)
                scores.append(_yes_no_probability(llm, yes_id, no_id))
            return scores
