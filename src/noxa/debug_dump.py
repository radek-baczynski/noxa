from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from noxa.request_context import log_prefix, request_id_var
from noxa.schemas import ExtractedPage, Passage, ScoredPassage, SelectedPassage

logger = logging.getLogger(__name__)

META_FILENAME = "meta.md"


def dump_root(base: Path | str = "dump") -> Path:
    root = Path(base)
    root.mkdir(parents=True, exist_ok=True)
    return root


def meta_path_for(dump_dir: Path) -> Path:
    return dump_dir / META_FILENAME


def _timestamp_dir_name(endpoint: str) -> str:
    ts = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    safe = endpoint.strip("/").replace("/", "_") or "request"
    return f"{safe}_{ts}"


def passage_dict(p: Passage) -> dict[str, Any]:
    return {
        "passage_id": p.passage_id,
        "source_id": p.source_id,
        "url": p.url,
        "title": p.title,
        "text": p.text,
        "token_count": p.token_count,
        "source_rank": p.source_rank,
        "start_char": p.start_char,
        "end_char": p.end_char,
    }


def scored_passage_dict(sp: ScoredPassage) -> dict[str, Any]:
    return {
        **passage_dict(sp.passage),
        "bm25_score": sp.bm25_score,
        "embedding_score": sp.embedding_score,
        "merged_score": sp.merged_score,
        "rerank_score": sp.rerank_score,
    }


def page_dict(page: ExtractedPage) -> dict[str, Any]:
    data = page.model_dump()
    data["html_chars"] = len(page.html or "")
    data["cleaned_html_chars"] = len(page.cleaned_html or "")
    data.pop("html", None)
    data.pop("cleaned_html", None)
    return data


def write_page_html_files(
    dumper: "DebugDumper",
    pages: list[ExtractedPage],
    stage: str = "fetch",
    sources: list[str] | None = None,
) -> list[str]:
    """Write raw and cleaned HTML sidecar files; return paths written."""
    written: list[str] = []
    index: list[dict[str, Any]] = []
    for i, page in enumerate(pages, start=1):
        source = sources[i - 1] if sources and i - 1 < len(sources) else "unknown"
        entry: dict[str, Any] = {
            "index": i,
            "url": page.url,
            "final_url": page.final_url,
            "title": page.title,
            "source": source,
            "raw_html": None,
            "cleaned_html": None,
        }
        if page.html:
            name = f"pages_html/page_{i}_raw.html"
            dumper.write_text(name, page.html, stage=stage)
            entry["raw_html"] = name
            written.append(name)
        if page.cleaned_html:
            name = f"pages_html/page_{i}_cleaned.html"
            dumper.write_text(name, page.cleaned_html, stage=stage)
            entry["cleaned_html"] = name
            written.append(name)
        index.append(entry)
    dumper.write_json("pages_html/index.json", index, stage=stage)
    written.append("pages_html/index.json")
    return written


def write_page_cache_files(
    dumper: "DebugDumper",
    records: list[dict[str, Any]],
    stage: str = "fetch",
) -> list[str]:
    """Dump SQLite cache entries (hits and freshly stored) for analysis."""
    written: list[str] = []
    index: list[dict[str, Any]] = []
    for i, rec in enumerate(records, start=1):
        stored = rec.get("stored") or {}
        files: list[str] = []
        entry: dict[str, Any] = {
            "index": i,
            "url": rec["url"],
            "cache_key": rec.get("cache_key"),
            "cache_hit": rec.get("cache_hit", False),
            "fetched_this_request": rec.get("fetched_this_request", False),
            "files": files,
        }
        if stored:
            meta = {
                k: v
                for k, v in stored.items()
                if k not in ("html", "cleaned_html", "text")
            }
            meta["text_chars"] = len(stored.get("text") or "")
            meta["html_chars"] = len(stored.get("html") or "")
            meta["cleaned_html_chars"] = len(stored.get("cleaned_html") or "")
            meta_name = f"pages_cache/page_{i}_meta.json"
            dumper.write_json(meta_name, meta, stage=stage)
            files.append(meta_name)
            if stored.get("text"):
                name = f"pages_cache/page_{i}_text.md"
                dumper.write_text(name, stored["text"], stage=stage)
                files.append(name)
            if stored.get("html"):
                name = f"pages_cache/page_{i}_raw.html"
                dumper.write_text(name, stored["html"], stage=stage)
                files.append(name)
            if stored.get("cleaned_html"):
                name = f"pages_cache/page_{i}_cleaned.html"
                dumper.write_text(name, stored["cleaned_html"], stage=stage)
                files.append(name)
        else:
            miss_name = f"pages_cache/page_{i}_miss.json"
            dumper.write_json(
                miss_name,
                {"url": rec["url"], "cache_hit": False, "note": "not in cache"},
                stage=stage,
            )
            files.append(miss_name)
        entry["files"] = files
        index.append(entry)
        written.extend(files)
    dumper.write_json("pages_cache/index.json", index, stage=stage)
    written.append("pages_cache/index.json")
    return written


def selected_passage_dict(s: SelectedPassage) -> dict[str, Any]:
    return s.model_dump()


@dataclass
class StageRecord:
    name: str
    latency_ms: int | None = None
    notes: str = ""
    files: list[str] = field(default_factory=list)


@dataclass
class DebugDumper:
    endpoint: str
    request: dict[str, Any]
    dump_base: Path
    request_id: str | None = None
    dir: Path = field(init=False)
    created_at: datetime = field(init=False)
    stages: list[StageRecord] = field(default_factory=list)
    _file_index: list[tuple[str, str, str]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.request_id = self.request_id or request_id_var.get()
        self.created_at = datetime.now()
        self.dir = dump_root(self.dump_base) / _timestamp_dir_name(self.endpoint)
        self.dir.mkdir(parents=True, exist_ok=True)
        self._write_stub_meta()

    @property
    def meta_file(self) -> Path:
        return meta_path_for(self.dir)

    def _write_stub_meta(self) -> None:
        """Placeholder hub file while the pipeline is still writing artifacts."""
        lines = [
            f"# Noxa debug dump: `{self.endpoint}`",
            "",
            "> **Start here.** This file is the index for all dump artifacts.",
            "",
            "Status: **in progress** — files are being written.",
            "",
            "## Summary",
            "",
            f"| Field | Value |",
            f"|-------|-------|",
            f"| Endpoint | `{self.endpoint}` |",
            f"| Request ID | `{self.request_id or 'n/a'}` |",
            f"| Started | {self.created_at.isoformat(timespec='seconds')} |",
            f"| Directory | `{self.dir.name}/` |",
            "",
            "Refresh this file when the request completes (see finalized `meta.md`).",
            "",
        ]
        self.meta_file.write_text("\n".join(lines), encoding="utf-8")

    def write_json(self, filename: str, data: Any, stage: str | None = None) -> Path:
        path = self.dir / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        self._file_index.append((filename, "json", stage or ""))
        return path

    def write_text(self, filename: str, text: str, stage: str | None = None) -> Path:
        path = self.dir / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        self._file_index.append((filename, "text", stage or ""))
        return path

    def add_stage(
        self,
        name: str,
        latency_ms: int | None = None,
        notes: str = "",
        files: list[str] | None = None,
    ) -> None:
        self.stages.append(
            StageRecord(
                name=name,
                latency_ms=latency_ms,
                notes=notes,
                files=files or [],
            )
        )

    def _build_meta_md(
        self,
        *,
        mode_config: dict[str, Any] | None,
        timing: dict[str, Any] | None,
        model_infer: dict[str, Any] | None,
        extra_meta: dict[str, Any] | None,
    ) -> str:
        query = self.request.get("query", self.request.get("urls", ""))
        total_ms = timing.get("total") if timing else None

        lines = [
            f"# Noxa debug dump: `{self.endpoint}`",
            "",
            "> **Start here.** This file is the center of the dump — parameters, "
            "pipeline stages, and links to every raw/processed artifact.",
            "",
            "## Table of contents",
            "",
            "- [Summary](#summary)",
            "- [Quick links](#quick-links)",
            "- [Request parameters](#request-parameters)",
        ]
        if mode_config:
            lines.append("- [Mode configuration](#mode-configuration)")
        if timing:
            lines.append("- [Timing](#timing-ms)")
        if model_infer:
            lines.append("- [Model inference](#model-inference-ms)")
        lines.append("- [Pipeline stages](#pipeline-stages)")
        lines.append("- [Complete file index](#complete-file-index)")
        if extra_meta:
            lines.append("- [Additional metadata](#additional-metadata)")
        lines.append("")

        lines.extend(
            [
                "## Summary",
                "",
                "| Field | Value |",
                "|-------|-------|",
                f"| Endpoint | `{self.endpoint}` |",
                f"| Request ID | `{self.request_id or 'n/a'}` |",
                f"| Dump folder | `{self.dir.name}/` |",
                f"| Started | {self.created_at.isoformat(timespec='seconds')} |",
                f"| Finalized | {datetime.now().isoformat(timespec='seconds')} |",
            ]
        )
        if query:
            lines.append(f"| Query | {query!r} |")
        if total_ms is not None:
            lines.append(f"| Total latency | **{total_ms} ms** |")
        lines.append(f"| Artifact count | {len(self._file_index)} files |")
        lines.append("")

        lines.extend(
            [
                "## Quick links",
                "",
                "Core artifacts:",
                "",
                f"- [{META_FILENAME}](./{META_FILENAME}) — this file",
                f"- [request.json](./request.json) — API request body",
            ]
        )
        if (self.dir / "response.json").exists() or any(
            f[0] == "response.json" for f in self._file_index
        ):
            lines.append(f"- [response.json](./response.json) — API response body")
        if mode_config:
            lines.append(f"- [mode_config.json](./mode_config.json) — mode preset")
        if timing:
            lines.append(f"- [timing.json](./timing.json) — stage timings")
        if model_infer:
            lines.append(
                f"- [model_infer.json](./model_infer.json) — model-only infer time & calls"
            )
        lines.append("")

        lines.extend(
            [
                "## Request parameters",
                "",
                "See [request.json](./request.json).",
                "",
                "```json",
                json.dumps(self.request, indent=2, ensure_ascii=False, default=str),
                "```",
                "",
            ]
        )

        if mode_config:
            lines.extend(
                [
                    "## Mode configuration",
                    "",
                    "See [mode_config.json](./mode_config.json).",
                    "",
                    "```json",
                    json.dumps(mode_config, indent=2, default=str),
                    "```",
                    "",
                ]
            )

        if timing:
            lines.extend(
                [
                    "## Timing (ms)",
                    "",
                    "See [timing.json](./timing.json).",
                    "",
                    "| Stage | ms |",
                    "|-------|-----|",
                ]
            )
            for key, val in timing.items():
                if val is not None and key != "total":
                    lines.append(f"| {key} | {val} |")
            if total_ms is not None:
                lines.append(f"| **total** | **{total_ms}** |")
            lines.append("")

        if model_infer:
            lines.extend(
                [
                    "## Model inference (ms)",
                    "",
                    "Wall-clock time and item counts for **model forward passes only** "
                    "(excludes cache hits, KNN, load, bootstrap).",
                    "",
                    "See [model_infer.json](./model_infer.json).",
                    "",
                    "| Role | time (ms) | calls |",
                    "|------|-----------|-------|",
                    f"| embedding | {model_infer.get('embedding_time', 0)} | {model_infer.get('embedding_calls', 0)} |",
                    f"| rerank | {model_infer.get('rerank_time', 0)} | {model_infer.get('rerank_calls', 0)} |",
                    f"| answer | {model_infer.get('answer_time', 0)} | {model_infer.get('answer_calls', 0)} |",
                    "",
                ]
            )
            answer_llm = model_infer.get("answer_llm")
            if answer_llm:
                lines.extend(
                    [
                        "### Answer LLM throughput",
                        "",
                        "| Phase | ms | tokens | tok/s |",
                        "|-------|-----|--------|-------|",
                        f"| prefill | {answer_llm.get('prefill_ms', 0)} | {answer_llm.get('prefill_tokens', 0)} | {answer_llm.get('prefill_tps', 0)} |",
                        f"| decode | {answer_llm.get('decode_ms', 0)} | {answer_llm.get('decode_tokens', 0)} | {answer_llm.get('decode_tps', 0)} |",
                        "",
                    ]
                )

        lines.extend(["## Pipeline stages", ""])
        for i, stage in enumerate(self.stages, start=1):
            lat = f"{stage.latency_ms}ms" if stage.latency_ms is not None else "—"
            anchor = stage.name.lower().replace(" ", "-")
            lines.append(f"### {i}. {stage.name} ({lat}) {{#{anchor}}}")
            if stage.notes:
                lines.append("")
                if isinstance(stage.notes, dict):
                    lines.append("```json")
                    lines.append(json.dumps(stage.notes, indent=2, default=str))
                    lines.append("```")
                else:
                    lines.append(str(stage.notes))
            if stage.files:
                lines.append("")
                lines.append("Artifacts:")
                for f in stage.files:
                    lines.append(f"- [{f}](./{f})")
            lines.append("")

        lines.extend(["## Complete file index", ""])
        lines.append(
            "All files written under this dump (excluding this index until finalized)."
        )
        lines.append("")
        lines.append("| File | Type | Stage |")
        lines.append("|------|------|-------|")
        for filename, kind, stage in sorted(self._file_index):
            if filename == META_FILENAME:
                continue
            stage_label = stage or "—"
            lines.append(f"| [{filename}](./{filename}) | {kind} | {stage_label} |")
        lines.append("")
        lines.append(
            f"| [{META_FILENAME}](./{META_FILENAME}) | markdown | index |"
        )
        lines.append("")

        if extra_meta:
            lines.extend(
                [
                    "## Additional metadata",
                    "",
                    "```json",
                    json.dumps(extra_meta, indent=2, default=str),
                    "```",
                    "",
                ]
            )

        return "\n".join(lines)

    def finalize(
        self,
        *,
        mode_config: dict[str, Any] | None = None,
        timing: dict[str, Any] | None = None,
        model_infer: dict[str, Any] | None = None,
        response: dict[str, Any] | None = None,
        extra_meta: dict[str, Any] | None = None,
    ) -> Path:
        self.write_json("request.json", self.request, stage="request")
        if mode_config is not None:
            self.write_json("mode_config.json", mode_config, stage="config")
        if timing is not None:
            self.write_json("timing.json", timing, stage="timing")
        if model_infer is not None:
            self.write_json("model_infer.json", model_infer, stage="timing")
        if response is not None:
            self.write_json("response.json", response, stage="response")

        meta_body = self._build_meta_md(
            mode_config=mode_config,
            timing=timing,
            model_infer=model_infer,
            extra_meta=extra_meta,
        )
        self.meta_file.write_text(meta_body, encoding="utf-8")
        self._file_index.append((META_FILENAME, "markdown", "index"))

        logger.info(
            "%sdebug dump written to %s (hub: %s)",
            log_prefix(),
            self.dir,
            self.meta_file,
        )
        return self.dir
