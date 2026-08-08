from __future__ import annotations

import asyncio
from pathlib import Path
from urllib.parse import urlparse

from storysmith.settings import Settings


class LocalStorage:
    """StoragePort backed by the local filesystem. Dev/tests only."""

    def __init__(self, settings: Settings) -> None:
        self._root = Path(settings.output_dir)

    async def put(self, *, key: str, data: bytes, content_type: str) -> str:
        path = self._root / key
        await asyncio.to_thread(self._write, path, data)
        return f"file://{path.resolve()}"

    async def get(self, *, uri: str) -> bytes:
        path = self._uri_to_path(uri)
        return await asyncio.to_thread(path.read_bytes)

    async def presign(self, *, uri: str, expires_s: int = 3600) -> str:
        return uri

    @staticmethod
    def _write(path: Path, data: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

    @staticmethod
    def _uri_to_path(uri: str) -> Path:
        parsed = urlparse(uri)
        if parsed.scheme != "file":
            raise ValueError(f"LocalStorage cannot resolve non-file uri: {uri!r}")
        return Path(parsed.path)
