from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GgufArtifact:
    repo_id: str
    filename: str
