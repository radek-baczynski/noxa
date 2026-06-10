from __future__ import annotations

import logging
import re

from huggingface_hub import list_repo_files

from noxa.runtime.artifacts import GgufArtifact

logger = logging.getLogger(__name__)


def gguf_artifact_from_repo(
    repo_id: str,
    *,
    quant: str = "Q4_K_M",
    token: str | None = None,
) -> GgufArtifact:
    """Pick a quantized .gguf file from a Hugging Face repo (no Hub search)."""
    quant_re = re.compile(re.escape(quant), re.IGNORECASE)
    files = sorted(
        f
        for f in list_repo_files(repo_id, token=token)
        if f.endswith(".gguf") and quant_re.search(f)
    )
    if not files:
        raise FileNotFoundError(
            f"No {quant} .gguf file found in Hugging Face repo {repo_id!r}"
        )
    artifact = GgufArtifact(repo_id=repo_id, filename=files[0])
    logger.info(
        "GGUF repo %s quant %s -> %s",
        repo_id,
        quant,
        artifact.filename,
    )
    return artifact
