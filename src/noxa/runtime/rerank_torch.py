from __future__ import annotations

import time
from functools import lru_cache

from noxa.model_timing import elapsed_ms, log_model, timed_model_op
from noxa.runtime.detect import infer_device
from noxa.config import Settings
from noxa.runtime.manifest import resolve_artifact

INSTRUCTION = (
    "Given a web search query, retrieve relevant passages that answer the query"
)
RERANK_PASSAGE_MAX_CHARS = 512


def _truncate_passage(text: str, max_chars: int = RERANK_PASSAGE_MAX_CHARS) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "..."


@lru_cache(maxsize=1)
def _load_qwen_reranker(model_name: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = infer_device()
    log_model("load start", model_name, backend="torch", device=device)
    t0 = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        trust_remote_code=True,
    ).to(device)
    model.eval()
    token_true = tokenizer.convert_tokens_to_ids("yes")
    token_false = tokenizer.convert_tokens_to_ids("no")
    prefix = (
        "<|im_start|>system\n"
        "Judge whether the Document meets the requirements based on the Query and "
        'the Instruct provided. Note that the answer can only be "yes" or "no".'
        "\n"
        "<|im_start|>user\n"
        f"<Instruct>: {INSTRUCTION}\n"
    )
    suffix = "\n<|im_start|>assistant\n\n\n\n\n"
    log_model("load done", model_name, elapsed_ms(t0), backend="torch", device=device)
    return tokenizer, model, device, token_true, token_false, prefix, suffix


def _format_pair(prefix: str, suffix: str, query: str, passage: str) -> str:
    return f'{prefix}<Query>: {query}\n<Document>: {passage}{suffix}'


class TorchRerankBackend:
    backend_id = "torch"

    def __init__(self, settings: Settings | None = None) -> None:
        self.model_id = resolve_artifact("rerank", "model", settings)

    def score(self, query: str, passages: list[str]) -> list[float]:
        if not passages:
            return []
        import torch

        tokenizer, model, device, token_true, token_false, prefix, suffix = (
            _load_qwen_reranker(self.model_id)
        )
        texts = [
            _format_pair(prefix, suffix, query, _truncate_passage(p))
            for p in passages
        ]
        with timed_model_op(
            self.model_id,
            "infer rerank",
            infer_kind="rerank",
            infer_calls=len(passages),
            backend=self.backend_id,
            batch=len(passages),
        ):
            inputs = tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=8192,
                return_tensors="pt",
            ).to(device)
            with torch.no_grad():
                logits = model(**inputs).logits[:, -1, :]
            true_vec = logits[:, token_true]
            false_vec = logits[:, token_false]
            scores = torch.stack([false_vec, true_vec], dim=1)
            scores = torch.nn.functional.log_softmax(scores, dim=1)[:, 1].exp()
        return scores.tolist()
