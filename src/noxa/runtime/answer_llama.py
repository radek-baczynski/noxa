from __future__ import annotations

import time
from functools import lru_cache
from pathlib import Path

from noxa.answer_contract import build_answer_prompt, parse_answer_json
from noxa.llm_perf import perf_from_llama
from noxa.model_stats import get_model_infer_stats
from noxa.model_timing import elapsed_ms, log_model, timed_model_op
from noxa.runtime.detect import LlamaOffloadConfig
from noxa.runtime.interfaces import AnswerResult
from noxa.config import Settings
from noxa.runtime.artifacts import GgufArtifact
from noxa.runtime.manifest import resolve_gguf_artifact
from noxa.schemas import Citation


@lru_cache(maxsize=4)
def _load_llama(
    repo_id: str,
    filename: str,
    offload_key: str,
    cache_dir: str,
    token: str | None,
) -> tuple[object, str]:
    from llama_cpp import Llama

    artifact = GgufArtifact(repo_id=repo_id, filename=filename)
    model_path = _resolve_gguf_path(artifact, Path(cache_dir), token=token)

    parts = offload_key.split(",")
    n_gpu_layers = int(parts[0])
    n_threads = int(parts[1])
    n_ctx = int(parts[2]) if len(parts) > 2 else 4096

    log_model("load start", str(model_path), backend="llama_cpp", repo=repo_id)
    t0 = time.perf_counter()
    llm = Llama(
        model_path=str(model_path),
        n_ctx=n_ctx,
        n_gpu_layers=n_gpu_layers,
        n_threads=n_threads,
        verbose=False,
    )
    log_model("load done", filename, elapsed_ms(t0), backend="llama_cpp")
    return llm, filename


def _resolve_gguf_path(
    artifact: GgufArtifact,
    cache_dir: Path,
    *,
    token: str | None = None,
) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    local = cache_dir / artifact.filename
    if local.exists():
        return local
    from huggingface_hub import hf_hub_download

    downloaded = hf_hub_download(
        repo_id=artifact.repo_id,
        filename=artifact.filename,
        local_dir=str(cache_dir),
        token=token,
    )
    return Path(downloaded)


class LlamaAnswerBackend:
    backend_id = "llama_cpp"

    def __init__(
        self,
        role: str,
        offload: LlamaOffloadConfig,
        model_cache_dir: str,
        hf_token: str | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.role = role
        self.offload = offload
        self.model_cache_dir = model_cache_dir
        self.hf_token = hf_token
        self.settings = settings
        self.artifact = resolve_gguf_artifact(
            role, model_cache_dir, token=hf_token, settings=settings
        )
        self.model_id = self.artifact.filename

    def _offload_key(self) -> str:
        return f"{self.offload.n_gpu_layers},{self.offload.n_threads},{self.offload.n_ctx}"

    def generate(
        self,
        query: str,
        documents: list[dict[str, str]],
        max_output_tokens: int,
    ) -> AnswerResult:
        t_total = time.perf_counter()
        log_model(
            "infer start",
            self.model_id,
            backend=self.backend_id,
            role=self.role,
            repo=self.artifact.repo_id,
            docs=len(documents),
        )
        llm, _ = _load_llama(
            self.artifact.repo_id,
            self.artifact.filename,
            self._offload_key(),
            self.model_cache_dir,
            self.hf_token,
        )
        prompt = build_answer_prompt(query, documents)

        import llama_cpp

        llama_cpp.llama_perf_context_reset(llm.ctx)  # type: ignore[attr-defined]

        with timed_model_op(
            self.model_id,
            "infer generate",
            infer_kind="answer",
            infer_calls=1,
            backend=self.backend_id,
            max_new_tokens=max_output_tokens,
        ):
            out = llm.create_completion(
                prompt=prompt,
                max_tokens=max_output_tokens,
                temperature=0.0,
                top_p=1.0,
                repeat_penalty=1.1,
            )
        llm_perf = perf_from_llama(llm, out.get("usage"))
        get_model_infer_stats().set_answer_llm(llm_perf)
        log_model(
            "infer perf",
            self.model_id,
            backend=self.backend_id,
            prefill_ms=llm_perf.prefill_ms,
            prefill_tps=llm_perf.prefill_tps,
            decode_ms=llm_perf.decode_ms,
            decode_tps=llm_perf.decode_tps,
            prompt_tokens=llm_perf.prompt_tokens,
            completion_tokens=llm_perf.completion_tokens,
        )
        raw = out["choices"][0]["text"]
        parsed = parse_answer_json(raw)
        citations = [
            Citation(source_id=i, url=None, title=None, supports=None)
            for i in parsed["citations"]
        ]
        result = AnswerResult(
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
        log_model(
            "infer done",
            self.model_id,
            elapsed_ms(t_total),
            backend=self.backend_id,
            abstained=result.abstained,
        )
        return result
