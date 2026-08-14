from __future__ import annotations

import asyncio
from pathlib import Path

from storysmith.settings import Settings

_SCHEME = "local://"


class LocalStorage:
    """StoragePort backed by the local filesystem. Dev/tests only.

    Stores the *relative* key in the URI (local://{key}), resolved against
    this process's own settings.output_dir at read time -- not an absolute
    path baked in at write time. An absolute path is only ever meaningful to
    the filesystem that wrote it: a scene_stills call running inside a
    Docker container writes under /app/out, and a later videographer call
    resuming the same project from the host process (different absolute
    root, e.g. via a shared bind mount) would get a FileNotFoundError trying
    to open that literal /app/out/... path. Relative keys resolve correctly
    in both places as long as they share the same underlying directory.
    """

    def __init__(self, settings: Settings) -> None:
        self._root = Path(settings.output_dir)

    async def put(self, *, key: str, data: bytes, content_type: str) -> str:
        path = self._root / key
        await asyncio.to_thread(self._write, path, data)
        return f"{_SCHEME}{key}"

    async def get(self, *, uri: str) -> bytes:
        path = self._uri_to_path(uri)
        return await asyncio.to_thread(path.read_bytes)

    async def presign(self, *, uri: str, expires_s: int = 3600) -> str:
        return uri

    @staticmethod
    def _write(path: Path, data: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

    def _uri_to_path(self, uri: str) -> Path:
        if not uri.startswith(_SCHEME):
            raise ValueError(f"LocalStorage cannot resolve non-local uri: {uri!r}")
        key = uri[len(_SCHEME) :]
        return self._root / key
