from __future__ import annotations

import hashlib


def sha256_hex(*parts: str) -> str:
    """Deterministic content hash of the given string parts, joined in order."""
    h = hashlib.sha256()
    for part in parts:
        h.update(part.encode("utf-8"))
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()
