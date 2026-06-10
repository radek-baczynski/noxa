from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from typing import Any, Iterator

from noxa.model_stats import InferKind, get_model_infer_stats
from noxa.request_context import log_prefix

logger = logging.getLogger("noxa.model")


def elapsed_ms(t0: float) -> int:
    return int((time.perf_counter() - t0) * 1000)


def log_model(
    event: str,
    model: str,
    ms: int | None = None,
    **fields: Any,
) -> None:
    parts = [f"{log_prefix()}model {event}", f"model={model!r}"]
    if ms is not None:
        parts.append(f"latency={ms}ms")
    for key, value in fields.items():
        parts.append(f"{key}={value}")
    logger.info(" ".join(parts))


@contextmanager
def timed_model_op(
    model: str,
    op: str,
    *,
    infer_kind: InferKind | None = None,
    infer_calls: int = 1,
    **fields: Any,
) -> Iterator[None]:
    t0 = time.perf_counter()
    try:
        yield
    finally:
        ms = elapsed_ms(t0)
        log_model(f"{op} done", model, ms, **fields)
        if infer_kind is not None:
            get_model_infer_stats().record(infer_kind, ms, infer_calls)
