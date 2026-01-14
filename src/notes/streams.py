from __future__ import annotations

import re
from typing import Optional


_STREAM_SAFE = re.compile(r"[^a-z0-9/_-]+")
_STREAM_SLASHES = re.compile(r"/+")
_STREAM_WS = re.compile(r"\s+")


def normalize_stream(value: Optional[str], default: str = "inbox") -> str:
    """
    Normalize a stream name to a stable, path-like identifier.
    - Lowercase
    - '/' separators
    - Whitespace -> '-'
    - Drops unsupported chars
    """
    if not value:
        return default

    stream = value.strip().lower().replace("\\", "/")
    stream = _STREAM_WS.sub("-", stream)
    stream = _STREAM_SAFE.sub("-", stream)
    stream = _STREAM_SLASHES.sub("/", stream)
    stream = stream.strip("/-")
    return stream or default


def stream_matches(candidate: str, filter_stream: Optional[str]) -> bool:
    """
    Returns True when candidate is within filter_stream's subtree.
    - filter_stream="work" matches "work" and "work/bh"
    - filter_stream=None matches everything
    """
    if not filter_stream:
        return True
    filter_norm = normalize_stream(filter_stream)
    candidate_norm = normalize_stream(candidate)
    return candidate_norm == filter_norm or candidate_norm.startswith(filter_norm + "/")

