#!/usr/bin/env python3
"""Benchmark llama.cpp answer backends across roles on a fixed query/passage set."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from noxa.config import get_settings
from noxa.runtime.registry import RuntimeRegistry


def _load_fixture(path: Path) -> tuple[str, list[dict[str, str]]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    query = data.get("query") or data["request"]["query"]
    documents = data.get("documents") or data.get("answer_documents")
    if documents is None:
        selected = data.get("selected_passages") or []
        by_source: dict[str, list[str]] = {}
        for item in selected:
            sid = item.get("source_id", "src")
            by_source.setdefault(sid, []).append(item["text"])
        documents = [
            {"text": "\n\n".join(parts)} for parts in by_source.values()
        ]
    return query, documents


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark Noxa answer backends")
    parser.add_argument(
        "--fixture",
        type=Path,
        default=ROOT / "dump",
        help="Debug dump dir or JSON with query/documents",
    )
    parser.add_argument(
        "--roles",
        default="answer_fast,answer_default",
        help="Comma-separated model roles",
    )
    parser.add_argument("--output", type=Path, default=Path("benchmark_answer.csv"))
    parser.add_argument("--max-output-tokens", type=int, default=256)
    args = parser.parse_args()

    fixture_path = args.fixture
    if fixture_path.is_dir():
        docs_file = fixture_path / "answer_documents.json"
        if not docs_file.exists():
            raise SystemExit(f"No answer_documents.json in {fixture_path}")
        if (fixture_path / "request.json").exists():
            query = json.loads(
                (fixture_path / "request.json").read_text(encoding="utf-8")
            ).get("query", "benchmark query")
        else:
            query = "benchmark query"
        documents = json.loads(docs_file.read_text(encoding="utf-8"))
    else:
        query, documents = _load_fixture(fixture_path)

    registry = RuntimeRegistry.from_settings(get_settings())
    rows: list[dict[str, object]] = []

    for role in args.roles.split(","):
        role = role.strip()
        if not role:
            continue
        answer_backend = registry.answer_for_role(role)
        t0 = time.perf_counter()
        result = answer_backend.generate(
            query,
            documents,
            max_output_tokens=args.max_output_tokens,
        )
        elapsed = int((time.perf_counter() - t0) * 1000)
        rows.append(
            {
                "query": query,
                "backend": answer_backend.backend_id,
                "model": answer_backend.model_id,
                "role": role,
                "answer_ms": elapsed,
                "abstained": result.abstained,
                "citations": [c.source_id for c in result.citations],
                "answer_text": result.answer.replace("\n", " ")[:500],
            }
        )
        print(
            f"llama_cpp/{role}: {elapsed}ms abstained={result.abstained}",
            file=sys.stderr,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "query",
                "backend",
                "model",
                "role",
                "answer_ms",
                "abstained",
                "citations",
                "answer_text",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
