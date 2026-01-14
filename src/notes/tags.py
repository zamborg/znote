from __future__ import annotations

import re
from typing import Iterable, List, Optional


_TAG_WS = re.compile(r"\s+")
_TAG_SAFE = re.compile(r"[^a-z0-9_-]+")
_TAG_DASHES = re.compile(r"-+")


def normalize_tag(value: str) -> str:
    tag = (value or "").strip().lower()
    tag = _TAG_WS.sub("-", tag)
    tag = _TAG_SAFE.sub("-", tag)
    tag = _TAG_DASHES.sub("-", tag)
    tag = tag.strip("-")
    return tag


def normalize_tags(values: Optional[Iterable[str]]) -> List[str]:
    seen = set()
    out: List[str] = []
    for raw in values or []:
        tag = normalize_tag(raw)
        if not tag or tag in seen:
            continue
        seen.add(tag)
        out.append(tag)
    return out

