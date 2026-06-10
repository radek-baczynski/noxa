from __future__ import annotations

import json
import re

_JSON_BLOCK_RE = re.compile(r"\{[^{}]*\}", re.DOTALL)


def build_answer_prompt(query: str, documents: list[dict[str, str]]) -> str:
    """Build a portable JSON-answer prompt with numbered sources."""
    lines = [
        "Answer the question using ONLY the provided sources.",
        "Rules:",
        "- Use only facts from the sources.",
        "- Cite claims with source numbers like [1], [2].",
        '- If sources are insufficient, set "abstain": true and explain briefly.',
        "- Do not use outside knowledge.",
        '- Respond with a single JSON object: {"answer": str, "abstain": bool, "citations": [int], "confidence": "low"|"medium"|"high"}',
        "",
        f"Question: {query}",
        "",
        "Sources:",
    ]
    for i, doc in enumerate(documents, start=1):
        lines.append(f"[{i}] {doc.get('text', '')}")
    lines.extend(["", "JSON:"])
    return "\n".join(lines)


def parse_answer_json(raw: str) -> dict:
    """Parse model JSON output with salvage for markdown fences or extra text."""
    text = raw.strip()
    if "```" in text:
        parts = text.split("```")
        for part in parts:
            chunk = part.strip()
            if chunk.startswith("json"):
                chunk = chunk[4:].strip()
            if chunk.startswith("{"):
                text = chunk
                break

    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return _normalize_answer_dict(data)
    except json.JSONDecodeError:
        pass

    for match in _JSON_BLOCK_RE.finditer(raw):
        try:
            data = json.loads(match.group(0))
            if isinstance(data, dict) and "answer" in data:
                return _normalize_answer_dict(data)
        except json.JSONDecodeError:
            continue

    return {
        "answer": "",
        "abstain": True,
        "citations": [],
        "confidence": "low",
    }


def _normalize_answer_dict(data: dict) -> dict:
    answer = str(data.get("answer", "") or "").strip()
    abstain = bool(data.get("abstain", False))
    if not answer:
        abstain = True
    citations = data.get("citations") or []
    if not isinstance(citations, list):
        citations = []
    citations = [int(c) for c in citations if isinstance(c, (int, float, str)) and str(c).isdigit()]
    confidence = str(data.get("confidence", "medium") or "medium").lower()
    if confidence not in {"low", "medium", "high"}:
        confidence = "medium"
    return {
        "answer": answer,
        "abstain": abstain,
        "citations": sorted(set(citations)),
        "confidence": confidence,
    }
