from __future__ import annotations

from contextvars import ContextVar

request_id_var: ContextVar[str | None] = ContextVar("request_id", default=None)


def log_prefix() -> str:
    rid = request_id_var.get()
    return f"[{rid}] " if rid else ""
