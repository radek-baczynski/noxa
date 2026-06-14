from __future__ import annotations

import json
import re

from noxa.config import AnswerSize, answer_size_prompt_hint

_MALFORMED_CITATIONS_RE = re.compile(
    r'"citations"\s*:\s*(?:\[\d+\]\s*,\s*)+\[\d+\]',
)
_THINK_OPEN = "<" + "think" + ">"
_THINK_CLOSE = "</" + "think" + ">"
_THINKING_MARKERS = (
    _THINK_OPEN,
    _THINK_CLOSE,
    "<think>",
    "Thinking Process:",
    "\n\nWait,",
    "\n\nGiven the constraints",
    "\n\nHowever,",
    "\n\nAlternatively,",
    "\n\nSo the final answer",
)
_MIN_ANSWER_WORDS: dict[AnswerSize, int] = {
    AnswerSize.SHORT: 4,
    AnswerSize.MEDIUM: 8,
    AnswerSize.COMPREHENSIVE: 12,
}
_REQUIRED_ANSWER_KEYS = frozenset({"answer", "abstain", "citations"})

ANSWER_JSON_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "abstain": {"type": "boolean"},
        "citations": {"type": "array", "items": {"type": "integer"}},
        "confidence": {
            "type": "string",
            "enum": ["low", "medium", "high"],
        },
    },
    "required": ["answer", "abstain", "citations"],
    "additionalProperties": False,
}


def answer_json_schema_json() -> str:
    """JSON Schema string for llama.cpp grammar-constrained answer output."""
    return json.dumps(ANSWER_JSON_SCHEMA)


def select_documents_for_answer(
    query: str,
    documents: list[dict[str, str | int]],
) -> tuple[list[dict[str, str | int]], list[int]]:
    """Return documents for the LLM prompt; citation ids match source_id order."""
    del query
    if not documents:
        return documents, []
    source_ids = [int(doc.get("source_id", i + 1)) for i, doc in enumerate(documents)]
    return documents, source_ids


def remap_citation_source_ids(
    citations: list,
    prompt_source_ids: list[int],
) -> None:
    """Map 1-based prompt citation indices to AnswerSource ids."""
    for citation in citations:
        prompt_idx = citation.source_id
        if 1 <= prompt_idx <= len(prompt_source_ids):
            citation.source_id = prompt_source_ids[prompt_idx - 1]


def build_answer_messages(
    query: str,
    documents: list[dict[str, str]],
    *,
    answer_size: AnswerSize = AnswerSize.MEDIUM,
) -> list[dict[str, str]]:
    """Chat messages for Qwen/instruct models (preferred over flat completion)."""
    system = "\n".join(
        [
            "You answer questions using ONLY the numbered sources in the user message.",
            "Reply with one JSON object only. No markdown fences, no reasoning, no extra text.",
            "",
            "Rules:",
            "- Answer the exact question asked; do not substitute a different question.",
            "- Sources are numbered by relevance to the question: [1] is the most relevant.",
            "- If sources describe different same-named people or entities, answer ONLY about the one matching the question's qualifiers (website, company, role, place). Ignore mismatched sources even when they contain more detail.",
            "- Use only facts from the sources.",
            f"- {answer_size_prompt_hint(answer_size)}",
            "- Match answer length to the question; short factual questions need only a short cited answer.",
            "- Cite claims in the answer text with source numbers like [1], [2].",
            '- JSON "citations" must be a number array like [1, 2], not [1], [2].',
            '- If sources are insufficient, set "abstain": true and explain briefly in "answer".',
            "- Do not use outside knowledge.",
            'Required keys: "answer", "abstain", "citations". Optional: "confidence" (low|medium|high).',
        ]
    )
    source_lines = [
        f"[{i}] {doc.get('text', '')}" for i, doc in enumerate(documents, start=1)
    ]
    user = "\n".join([f"Question: {query}", "", "Sources:", *source_lines])
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def build_answer_prompt(
    query: str,
    documents: list[dict[str, str]],
    *,
    answer_size: AnswerSize = AnswerSize.MEDIUM,
) -> str:
    """Flat prompt text for debug dumps."""
    parts = [
        f"{message['role'].upper()}:\n{message['content']}"
        for message in build_answer_messages(query, documents, answer_size=answer_size)
    ]
    return "\n\n".join(parts)


def parse_answer_json(raw: str) -> dict:
    """Parse model JSON output with salvage for fences, thinking blocks, and bad citations."""
    parsed_candidates: list[tuple[int, dict]] = []
    for candidate in _json_candidates(raw):
        fixed = _fix_malformed_citations(candidate)
        try:
            data = json.loads(fixed)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict) and "answer" in data:
            parsed_candidates.append((_answer_json_score(data), data))

    if parsed_candidates:
        parsed_candidates.sort(key=lambda item: item[0], reverse=True)
        return _normalize_answer_dict(parsed_candidates[0][1])

    return {
        "answer": "",
        "abstain": True,
        "citations": [],
        "confidence": "low",
    }


def validate_answer_grounding(
    answer: str,
    documents: list[dict[str, str | int]],
    *,
    query: str = "",
    answer_size: AnswerSize = AnswerSize.MEDIUM,
) -> bool:
    """Reject empty, too-short, or low-overlap answers against source text."""
    del query
    if not answer.strip():
        return False

    normalized = answer.strip().strip("\"'")
    if len(normalized) < 12 or len(normalized.split()) < 3:
        return False

    corpus = " ".join(
        f"{doc.get('title', '')} {doc.get('text', '')}" for doc in documents
    ).lower()
    answer_l = answer.lower()

    words = [
        word
        for word in re.findall(r"[a-z]{4,}", answer_l)
        if word not in {"about", "their", "which", "through", "offers", "making", "with"}
    ]
    if not words:
        return len(answer.split()) >= 3

    hits = sum(1 for word in words[:14] if word in corpus)
    overlap_ok = hits >= max(1, len(words[:14]) // 5)
    if not overlap_ok:
        return False

    min_words = _MIN_ANSWER_WORDS.get(answer_size, 8)
    if len(answer.split()) < min_words:
        # Concise factual answers are valid when they ground in the sources.
        return len(answer.split()) >= 4
    return True


def _json_candidates(raw: str) -> list[str]:
    text = _strip_trailing_prose(_strip_model_artifacts(raw.strip()))
    candidates: list[str] = []

    def add(candidate: str | None) -> None:
        if not candidate:
            return
        chunk = candidate.strip()
        if chunk and chunk not in candidates:
            candidates.append(chunk)

    if "```" in text:
        for part in text.split("```"):
            chunk = part.strip()
            if chunk.startswith("json"):
                chunk = chunk[4:].strip()
            add(chunk if chunk.startswith("{") else _extract_first_json_object(chunk))

    add(text)
    add(_extract_first_json_object(text))
    for line in text.splitlines():
        add(_extract_first_json_object(line) or line.strip())

    return candidates


def _strip_trailing_prose(text: str) -> str:
    obj = _extract_first_json_object(text)
    if not obj:
        return text
    end = text.find(obj) + len(obj)
    rest = text[end:].lstrip()
    if rest and not rest.startswith("{"):
        return text[:end].strip()
    return text


def _strip_model_artifacts(text: str) -> str:
    lowered = text.lower()
    cut_at: int | None = None
    for marker in _THINKING_MARKERS:
        idx = lowered.find(marker.lower())
        if idx >= 0:
            cut_at = idx if cut_at is None else min(cut_at, idx)
    if cut_at is not None:
        text = text[:cut_at]
    return text.strip()


def _fix_malformed_citations(text: str) -> str:
    def repl(match: re.Match[str]) -> str:
        nums = re.findall(r"\[(\d+)\]", match.group(0))
        return f'"citations": [{", ".join(nums)}]'

    return _MALFORMED_CITATIONS_RE.sub(repl, text)


def _extract_first_json_object(text: str) -> str | None:
    start = text.find("{")
    if start < 0:
        return None

    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def _answer_json_score(data: dict) -> int:
    score = 0
    if _REQUIRED_ANSWER_KEYS.issubset(data.keys()):
        score += 4
    elif "citations" in data:
        score += 2
    citations = data.get("citations") or []
    if isinstance(citations, list) and citations:
        score += 2
    if data.get("confidence") in {"low", "medium", "high"}:
        score += 1
    return score


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
