from __future__ import annotations

from typing import Protocol

from pydantic import BaseModel

# SPEC-GAP: spec shows `from storysmith.models import *`; explicit imports used
# instead so ruff's F403 (wildcard import) and mypy --strict both pass cleanly.
from storysmith.models import Mode


class LLMPort(Protocol):
    async def complete_structured(
        self,
        *,
        system: str,
        user: str,
        schema: type[BaseModel],
        model_tier: str = "standard",  # "standard" | "vision"
        images: list[bytes] | None = None,
    ) -> tuple[BaseModel, float]: ...  # (parsed object, cost_usd)


class ImageGenPort(Protocol):
    async def generate(
        self, *, prompt: str, aspect_ratio: str, reference_image: bytes | None = None
    ) -> tuple[bytes, float]: ...

    # reference_image (Amendment 03): image-conditioned generation (e.g.
    # flux-kontext-pro) instead of pure text-to-image, to keep a character's
    # appearance consistent across independently-generated scene stills --
    # a stateless text-to-image model has no memory between calls, so
    # naming/describing a character in text alone renders a different-
    # looking character almost every time (confirmed live). None (the
    # default) preserves today's pure text-to-image behavior unchanged.


class VideoGenPort(Protocol):
    async def generate(
        self,
        *,
        prompt: str,
        duration_s: float,
        aspect_ratio: str,
        reference_image: bytes | None,
    ) -> tuple[bytes, float]: ...  # (mp4 bytes, cost_usd)


class MusicGenPort(Protocol):
    async def generate(
        self,
        *,
        mode: Mode,
        lyrics: str | None,
        description: str,
        duration_s: float,
    ) -> tuple[bytes, float]: ...  # (wav/mp3 bytes, cost_usd)


class TTSPort(Protocol):
    async def speak(self, *, text: str, voice: str) -> tuple[bytes, float]: ...


class TranscribePort(Protocol):
    # SPEC-GAP: spec shows `-> tuple[list[dict], float]`; parameterized as
    # dict[str, str | float] (mypy --strict rejects bare `dict`) to match the
    # word-level timing shape: [{"word": str, "start": float, "end": float}]
    async def transcribe(self, *, audio: bytes) -> tuple[list[dict[str, str | float]], float]: ...


class StoragePort(Protocol):
    async def put(self, *, key: str, data: bytes, content_type: str) -> str: ...  # returns uri
    async def get(self, *, uri: str) -> bytes: ...
    async def presign(self, *, uri: str, expires_s: int = 3600) -> str: ...


class PublishPort(Protocol):
    async def upload(
        self, *, video: bytes, thumbnail: bytes, title: str, description: str, tags: list[str]
    ) -> str: ...  # video URL


class NotifyPort(Protocol):
    async def send(self, *, text: str, link: str | None = None) -> None: ...
