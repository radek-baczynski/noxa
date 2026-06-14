from __future__ import annotations

from noxa.config import Settings
from noxa.model_timing import timed_model_op
from noxa.runtime.detect import LlamaOffloadConfig
from noxa.runtime.interfaces import EmbedKind
from noxa.runtime.llama_common import load_llama, offload_key
from noxa.runtime.manifest import resolve_gguf_artifact

NOMIC_QUERY_PREFIX = "search_query: "
NOMIC_PASSAGE_PREFIX = "search_document: "


def _prefix_text(text: str, kind: EmbedKind) -> str:
    if kind == "query":
        return f"{NOMIC_QUERY_PREFIX}{text}"
    return f"{NOMIC_PASSAGE_PREFIX}{text}"


class LlamaEmbedBackend:
    backend_id = "llama_cpp"

    def __init__(
        self,
        offload: LlamaOffloadConfig,
        model_cache_dir: str,
        hf_token: str | None = None,
        batch_size: int = 64,
        settings: Settings | None = None,
    ) -> None:
        self.offload = offload
        self.model_cache_dir = model_cache_dir
        self.hf_token = hf_token
        self.batch_size = batch_size
        self.settings = settings
        self.artifact = resolve_gguf_artifact(
            "embed", model_cache_dir, token=hf_token, settings=settings
        )
        self.model_id = self.artifact.filename

    def _llm(self) -> object:
        llm, _ = load_llama(
            self.artifact.repo_id,
            self.artifact.filename,
            offload_key(self.offload),
            self.model_cache_dir,
            self.hf_token,
            "embed",
        )
        return llm

    def encode(self, texts: list[str], kind: EmbedKind) -> list[list[float]]:
        if not texts:
            return []
        llm = self._llm()
        prefixed = [_prefix_text(t, kind) for t in texts]
        vectors: list[list[float]] = []
        with timed_model_op(
            self.model_id,
            "infer encode",
            infer_kind="embedding",
            infer_calls=len(texts),
            backend=self.backend_id,
            count=len(texts),
            kind=kind,
        ):
            for i in range(0, len(prefixed), self.batch_size):
                batch = prefixed[i : i + self.batch_size]
                result = llm.create_embedding(batch)
                vectors.extend(item["embedding"] for item in result["data"])
        return vectors
