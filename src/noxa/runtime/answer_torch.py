from __future__ import annotations

import re
import time
from functools import lru_cache

from noxa.answer_contract import build_answer_prompt, parse_answer_json
from noxa.llm_perf import perf_from_torch
from noxa.model_stats import get_model_infer_stats
from noxa.model_timing import elapsed_ms, log_model, timed_model_op
from noxa.runtime.detect import infer_device
from noxa.runtime.interfaces import AnswerResult
from noxa.config import Settings
from noxa.runtime.manifest import resolve_artifact
from noxa.schemas import Citation


@lru_cache(maxsize=2)
def _load_instruct_model(model_name: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = infer_device()
    log_model("load start", model_name, backend="torch_json", device=device)
    t0 = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        trust_remote_code=True,
    ).to(device)
    model.eval()
    log_model("load done", model_name, elapsed_ms(t0), backend="torch_json", device=device)
    return tokenizer, model, device


class TorchJsonAnswerBackend:
    backend_id = "torch"

    def __init__(self, role: str, settings: Settings | None = None) -> None:
        self.role = role
        self.model_id = resolve_artifact(role, "torch_model", settings)

    def generate(
        self,
        query: str,
        documents: list[dict[str, str]],
        max_output_tokens: int,
    ) -> AnswerResult:
        import torch

        t_total = time.perf_counter()
        tokenizer, model, device = _load_instruct_model(self.model_id)
        prompt = build_answer_prompt(query, documents)
        inputs = tokenizer([prompt], return_tensors="pt").to(device)

        t_gen = time.perf_counter()
        with timed_model_op(
            self.model_id,
            "infer generate",
            infer_kind="answer",
            infer_calls=1,
            backend=self.backend_id,
        ):
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_output_tokens,
                    do_sample=False,
                    repetition_penalty=1.1,
                    eos_token_id=tokenizer.eos_token_id,
                )
        gen_ms = (time.perf_counter() - t_gen) * 1000
        prompt_tokens = int(inputs.input_ids.shape[1])
        completion_tokens = int(outputs.shape[1] - prompt_tokens)
        llm_perf = perf_from_torch(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_ms=gen_ms,
        )
        get_model_infer_stats().set_answer_llm(llm_perf)
        log_model(
            "infer perf",
            self.model_id,
            backend=self.backend_id,
            decode_ms=llm_perf.decode_ms,
            decode_tps=llm_perf.decode_tps,
            prompt_tokens=llm_perf.prompt_tokens,
            completion_tokens=llm_perf.completion_tokens,
        )
        raw = tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1] :],
            skip_special_tokens=True,
        )
        parsed = parse_answer_json(raw)
        citations = [
            Citation(source_id=i, url=None, title=None, supports=None)
            for i in parsed["citations"]
        ]
        return AnswerResult(
            answer=parsed["answer"],
            abstained=parsed["abstain"],
            citations=citations,
            confidence=parsed["confidence"],
            raw_response=raw,
            model_prompt=prompt,
            incomplete=not parsed["answer"] and not parsed["abstain"],
            backend=self.backend_id,
            llm_perf=llm_perf,
        )


def extract_citations_from_text(answer: str) -> list[Citation]:
    ids = sorted({int(n) for n in re.findall(r"\[(\d+)\]", answer)})
    return [Citation(source_id=i, url=None, title=None, supports=None) for i in ids]
