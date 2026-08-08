from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel
from storysmith.models import (
    CharacterRef,
    Mode,
    MusicCue,
    QAReport,
    QAVerdict,
    Scene,
    SceneManifest,
    StyleContract,
)

if TYPE_CHECKING:
    from storysmith.pipeline import PortBundle

STUB_COST_USD = 0.001

_TMP_DIR = Path(tempfile.gettempdir()) / "storysmith_stubs"
_TMP_DIR.mkdir(parents=True, exist_ok=True)
_STUB_VIDEO_PATH = _TMP_DIR / "stub_scene.mp4"
_STUB_AUDIO_PATH = _TMP_DIR / "stub_audio.mp3"


def _ensure_ffmpeg_asset(path: Path, *lavfi_args: str) -> bytes:
    """Generate a tiny real media file with ffmpeg once, then cache it on disk."""
    if not path.exists():
        subprocess.run(  # noqa: S603 - fixed local ffmpeg invocation, no user input
            ["ffmpeg", "-y", *lavfi_args, str(path)],
            check=True,
            capture_output=True,
        )
    return path.read_bytes()


def _stub_video_bytes() -> bytes:
    return _ensure_ffmpeg_asset(
        _STUB_VIDEO_PATH,
        "-f",
        "lavfi",
        "-i",
        "color=c=blue:s=1080x1920:d=1",
        "-c:v",
        "libx264",
        "-t",
        "1",
        "-pix_fmt",
        "yuv420p",
    )


def _stub_audio_bytes() -> bytes:
    return _ensure_ffmpeg_asset(
        _STUB_AUDIO_PATH,
        "-f",
        "lavfi",
        "-i",
        "anullsrc=r=24000:cl=mono",
        "-t",
        "1",
        "-q:a",
        "9",
    )


def _canned_style_contract() -> StyleContract:
    return StyleContract(
        art_style="soft 2D cutout animation, thick outlines",
        palette=["#FFAA00", "#00AAFF"],
        mood="cheerful",
        tempo_bpm=100,
        characters=[
            CharacterRef(
                name="Ducky",
                description="a small yellow duck with a red scarf, big friendly eyes",
            )
        ],
        pacing_rules="keep cuts short and punchy, one idea per scene",
        negative_terms=["weapons", "scary faces", "strangers"],
    )


def _canned_scene_manifest() -> SceneManifest:
    scenes = [
        Scene(
            index=i,
            duration_s=8,
            video_prompt=f"soft 2D cutout animation, Ducky the yellow duck, scene {i}",
            narration=f"one two three duck {i}",
        )
        for i in range(5)
    ]
    return SceneManifest(
        title="Counting Ducks",
        description="A cheerful counting song with Ducky the duck.",
        tags=["kids", "counting", "rhyme"],
        total_duration_s=40.0,
        lyrics="One two three four five, Ducky comes alive",
        music_cues=[MusicCue(description="upbeat intro", start_s=0, end_s=4)],
        scenes=scenes,
    )


def _canned_qa_report_pass() -> QAReport:
    return QAReport(
        scene_index=None,
        verdict=QAVerdict.PASS,
        scores={
            "style_adherence": 0.9,
            "character_consistency": 0.9,
            "visual_artifacts": 0.95,
            "kid_appeal": 0.9,
            "safety": 1.0,
        },
        safety_flags=[],
        critique="",
    )


_CANNED_FACTORIES = {
    "StyleContract": _canned_style_contract,
    "SceneManifest": _canned_scene_manifest,
    "QAReport": _canned_qa_report_pass,
}


class StubLLM:
    async def complete_structured(
        self,
        *,
        system: str,
        user: str,
        schema: type[BaseModel],
        model_tier: str = "standard",
        images: list[bytes] | None = None,
    ) -> tuple[BaseModel, float]:
        factory = _CANNED_FACTORIES.get(schema.__name__)
        if factory is None:
            raise KeyError(f"StubLLM has no canned object for schema {schema.__name__!r}")
        return factory(), STUB_COST_USD


class StubImageGen:
    async def generate(self, *, prompt: str, aspect_ratio: str) -> tuple[bytes, float]:
        return b"STUB_IMAGE_BYTES", STUB_COST_USD


class StubVideoGen:
    async def generate(
        self,
        *,
        prompt: str,
        duration_s: float,
        aspect_ratio: str,
        reference_image: bytes | None,
    ) -> tuple[bytes, float]:
        return _stub_video_bytes(), STUB_COST_USD


class StubMusicGen:
    async def generate(
        self,
        *,
        mode: Mode,
        lyrics: str | None,
        description: str,
        duration_s: float,
    ) -> tuple[bytes, float]:
        return _stub_audio_bytes(), STUB_COST_USD


class StubTTS:
    async def speak(self, *, text: str, voice: str) -> tuple[bytes, float]:
        return _stub_audio_bytes(), STUB_COST_USD


class StubTranscribe:
    async def transcribe(self, *, audio: bytes) -> tuple[list[dict[str, str | float]], float]:
        words = [
            {"word": "one", "start": 0.0, "end": 0.3},
            {"word": "two", "start": 0.3, "end": 0.6},
            {"word": "three", "start": 0.6, "end": 1.0},
        ]
        return words, STUB_COST_USD


class StubStorage:
    def __init__(self) -> None:
        self._store: dict[str, bytes] = {}

    async def put(self, *, key: str, data: bytes, content_type: str) -> str:
        uri = f"stub://{key}"
        self._store[uri] = data
        return uri

    async def get(self, *, uri: str) -> bytes:
        return self._store[uri]

    async def presign(self, *, uri: str, expires_s: int = 3600) -> str:
        return uri


class StubPublish:
    async def upload(
        self, *, video: bytes, thumbnail: bytes, title: str, description: str, tags: list[str]
    ) -> str:
        return f"https://stub.youtube.example/watch?v={abs(hash(title))}"


class StubNotify:
    def __init__(self) -> None:
        self.sent: list[tuple[str, str | None]] = []

    async def send(self, *, text: str, link: str | None = None) -> None:
        self.sent.append((text, link))


def stub_port_bundle() -> PortBundle:
    # Local import: storysmith-core's pipeline.py must not statically depend on
    # storysmith-adapters (adapters depends on core, not the reverse). This
    # factory is a dev/test convenience only, wired in at call time.
    from storysmith.pipeline import PortBundle

    return PortBundle(
        llm=StubLLM(),
        image_gen=StubImageGen(),
        video_gen=StubVideoGen(),
        music_gen=StubMusicGen(),
        tts=StubTTS(),
        transcribe=StubTranscribe(),
        storage=StubStorage(),
        publish=StubPublish(),
        notify=StubNotify(),
    )
