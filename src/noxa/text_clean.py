from __future__ import annotations

import re

_MD_LINK_RE = re.compile(r"\[([^\]]*)\]\([^)]*\)")
_IMG_LINK_RE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_EMPTY_LINK_RE = re.compile(r"\[\]\([^)]*\)")


def clean_page_markdown(text: str) -> str:
    """Strip markdown link targets and image links; keep anchor text."""
    text = _IMG_LINK_RE.sub("", text)
    text = _EMPTY_LINK_RE.sub("", text)
    text = _MD_LINK_RE.sub(lambda m: m.group(1), text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
