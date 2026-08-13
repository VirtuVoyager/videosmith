from __future__ import annotations

import hashlib
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


def _read_comment_tag(audio: bytes) -> str:
    with tempfile.NamedTemporaryFile(suffix=".mp3") as tmp:
        tmp.write(audio)
        tmp.flush()
        result = subprocess.run(  # noqa: S603
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format_tags=comment",
                "-of",
                "default=nw=1:nk=1",
                tmp.name,
            ],
            capture_output=True,
            text=True,
            check=True,
        )
    return result.stdout.strip()


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


def _stub_audio_bytes_for_text(text: str) -> bytes:
    """A real, valid 1s mp3 -- but with `text` embedded in its comment tag so
    StubTranscribe can read it back verbatim instead of returning fixed
    filler words. Without this, the Critic's WER check (§6) always fails
    against stub audio, since none of the stub generators actually encode
    real speech -- there'd be no way to exercise the pipeline's audio-QA
    happy path with stubs at all otherwise."""
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    path = _TMP_DIR / f"stub_audio_{digest}.mp3"
    return _ensure_ffmpeg_asset(
        path,
        "-f",
        "lavfi",
        "-i",
        "anullsrc=r=24000:cl=mono",
        "-t",
        "1",
        "-metadata",
        f"comment={text}",
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
    # gen_mode defaults to i2v (Amendment 01) -- scene_image_prompt is
    # required for those scenes, so the canned manifest carries one even
    # though StubLLM bypasses director.py's post-validation that would
    # otherwise enforce it.
    scenes = [
        Scene(
            index=i,
            duration_s=8,
            video_prompt=(
                f"soft art style, the duck bobs its head up and down, "
                f"camera remains static, scene {i}"
            ),
            scene_image_prompt=(
                f"soft 2D cutout animation, Ducky the yellow duck centered in frame, "
                f"plain background, scene {i}"
            ),
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
        # Only the rhyme-mode master gets transcribed/WER-checked (§6); the
        # topical bed track never is, so it can stay generic.
        if mode == Mode.RHYME and lyrics:
            return _stub_audio_bytes_for_text(lyrics), STUB_COST_USD
        return _stub_audio_bytes(), STUB_COST_USD


class StubTTS:
    async def speak(self, *, text: str, voice: str) -> tuple[bytes, float]:
        return _stub_audio_bytes_for_text(text), STUB_COST_USD


class StubTranscribe:
    async def transcribe(self, *, audio: bytes) -> tuple[list[dict[str, str | float]], float]:
        text = _read_comment_tag(audio)
        words = []
        start = 0.0
        for word in text.split():
            words.append({"word": word, "start": start, "end": start + 0.3})
            start += 0.3
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
