from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel
from storysmith.models import Mode, QAReport, SceneManifest, StyleContract
from storysmith.pipeline import PortBundle

from storysmith_adapters.stubs import (
    StubLLM,
    StubNotify,
    StubPublish,
    StubStorage,
    _stub_audio_bytes,
)

STUB_COST_USD = 0.0  # already-paid-for recorded content -- replaying it costs nothing


class _Sequence:
    """Replays a fixed list of pre-recorded values, one per call, in the
    order they were originally produced. Correct as long as a fresh run
    calls the underlying port method the same number of times, in the same
    order, as the run that was recorded (true for a fresh run down the same
    code path -- same show, same skip-guards) -- this is a "replay this one
    exact prior run" tool, not a general request-matching cassette."""

    def __init__(self, values: list[Any], *, label: str) -> None:
        self._values = values
        self._label = label
        self._i = 0

    def next(self) -> Any:
        if self._i >= len(self._values):
            raise IndexError(
                f"RecordedFixtures: no more recorded {self._label} "
                f"(exhausted after {len(self._values)} calls) -- this recording doesn't "
                "cover a run shaped like this one"
            )
        value = self._values[self._i]
        self._i += 1
        return value


def _sorted_files(directory: Path, pattern: str) -> list[bytes]:
    if not directory.exists():
        return []
    return [p.read_bytes() for p in sorted(directory.glob(pattern), key=lambda p: p.name)]


def _seed_avatars(style: StyleContract, avatars_dir: Path, storage: StubStorage) -> StyleContract:
    """Loads each character's real avatar bytes into `storage` under a
    fresh stub:// URI and returns a copy of `style` pointing at it --
    style.json was captured from the *original* run and carries that run's
    real local:// URI, which nothing in a fresh recorded_port_bundle() ever
    populates. Poking StubStorage's dict directly (rather than its async
    `put`) keeps this constructor synchronous."""
    if not avatars_dir.exists():
        return style
    updated = []
    for character in style.characters:
        avatar_path = avatars_dir / f"char_{character.name}.png"
        if not avatar_path.exists():
            updated.append(character)
            continue
        uri = f"stub://avatars/char_{character.name}.png"
        storage._store[uri] = avatar_path.read_bytes()  # noqa: SLF001 -- same-package, see docstring
        updated.append(character.model_copy(update={"image_uri": uri}))
    return style.model_copy(update={"characters": updated})


class RecordedFixtures:
    """Loads a directory of real generated content pulled from a live run
    (see tests/fixtures/recorded/*/README.md) and replays it through the
    same Port protocols the real adapters implement -- for manual offline
    dev/testing against realistic media without spending money or waiting
    on real APIs. Not used by the unit-test suite (which needs stubs.py's
    tiny synthetic assets to stay fast); see recorded_port_bundle() below.
    """

    def __init__(self, root: Path) -> None:
        self.root = root
        self.style: StyleContract | None = None
        self.manifest: SceneManifest | None = None
        self.qa_reports: list[QAReport] = []
        # Owned by this instance (not a fresh one per port) so a style whose
        # characters' image_uri gets rewritten below (§ avatar seeding)
        # stays resolvable: Critic fetches a scene's character-reference
        # image by URI (vision-QA's character_consistency check), and the
        # style.json this was captured from carries the *original run's*
        # local:// URI, which a fresh StubStorage was never seeded with.
        self.storage = StubStorage()

        style_path = root / "style.json"
        if style_path.exists():
            style = StyleContract.model_validate_json(style_path.read_text())
            self.style = _seed_avatars(style, root / "avatars", self.storage)

        manifest_path = root / "manifest.json"
        if manifest_path.exists():
            self.manifest = SceneManifest.model_validate_json(manifest_path.read_text())

        qa_path = root / "qa_reports.json"
        if qa_path.exists():
            self.qa_reports = [QAReport.model_validate(r) for r in json.loads(qa_path.read_text())]

        llm_calls: dict[str, _Sequence] = {}
        if self.style is not None:
            llm_calls["StyleContract"] = _Sequence([self.style], label="StyleContract call")
        if self.manifest is not None:
            llm_calls["SceneManifest"] = _Sequence([self.manifest], label="SceneManifest call")
        # Only per-scene QAReports are real LLM calls (critic.py's vision-tier
        # complete_structured, once per scene) -- the aggregate audio verdict
        # (scene_index=None) is computed from WER in code, never requested
        # via the LLM port, so it's excluded here even though it's kept in
        # self.qa_reports as part of the full historical record.
        per_scene_qa = [r for r in self.qa_reports if r.scene_index is not None]
        if per_scene_qa:
            llm_calls["QAReport"] = _Sequence(per_scene_qa, label="QAReport call")
        self._llm_calls = llm_calls

        # Public: not consumed by any Recorded* port class (see
        # RecordedImageGen's docstring) -- exposed for direct use by anyone
        # wanting to replay POST /shows's own avatar generation instead.
        self.avatars = _Sequence(_sorted_files(root / "avatars", "*.png"), label="avatar image")
        self._scene_stills = _Sequence(
            _sorted_files(root / "scene_stills", "*.png"), label="scene still image"
        )
        self._scene_videos = _Sequence(
            _sorted_files(root / "scene_videos", "*.mp4"), label="scene video"
        )
        self._narration = _Sequence(
            _sorted_files(root / "narration", "*.mp3"), label="narration audio"
        )
        audio_bed_path = root / "audio_bed.mp3"
        self._audio_bed = (
            _Sequence([audio_bed_path.read_bytes()], label="audio bed")
            if audio_bed_path.exists()
            else _Sequence([], label="audio bed")
        )


class RecordedLLM:
    """Replays recorded StyleContract/SceneManifest calls; falls back to
    StubLLM's canned (synthetic, always-passing) response for any schema
    this fixture doesn't have real recordings for -- QAReport, notably:
    Critic needs a per-scene verdict on every run, but nothing in this
    codebase captures real Critic output today (that would mean actually
    trusting a live vision-model judgment call, not just replaying bytes).
    This fallback is what makes recorded_port_bundle() usable to run a full
    graph end to end out of the box, real style/script/media throughout
    except Critic's verdicts."""

    def __init__(self, fixtures: RecordedFixtures) -> None:
        self._fixtures = fixtures
        self._fallback = StubLLM()

    async def complete_structured(
        self,
        *,
        system: str,
        user: str,
        schema: type[BaseModel],
        model_tier: str = "standard",
        images: list[bytes] | None = None,
    ) -> tuple[BaseModel, float]:
        sequence = self._fixtures._llm_calls.get(schema.__name__)
        if sequence is not None:
            try:
                return sequence.next(), STUB_COST_USD
            except IndexError:
                pass  # recorded calls for this schema exhausted -- fall through
        return await self._fallback.complete_structured(
            system=system, user=user, schema=schema, model_tier=model_tier, images=images
        )


class RecordedImageGen:
    """Replays scene stills (scene_stills node). Avatar images are
    deliberately NOT part of this sequence: they're generated once via
    `POST /shows`, a plain FastAPI handler outside the LangGraph pipeline
    entirely -- a frozen-show episode run's `char_refs` node always skips
    (every character already has image_uri), so `ImageGenPort.generate` is
    never called for avatars within a *pipeline run* replay. A recorded
    show's avatars (fixtures.avatars) are for someone wanting to replay
    `POST /shows` itself, a separate, rarer use case -- pull them directly
    off a RecordedFixtures instance rather than through a PortBundle."""

    def __init__(self, fixtures: RecordedFixtures) -> None:
        self._stills = fixtures._scene_stills

    async def generate(
        self, *, prompt: str, aspect_ratio: str, reference_image: bytes | None = None
    ) -> tuple[bytes, float]:
        del reference_image  # replaying real recorded bytes either way
        return self._stills.next(), STUB_COST_USD


class RecordedVideoGen:
    def __init__(self, fixtures: RecordedFixtures) -> None:
        self._videos = fixtures._scene_videos

    async def generate(
        self,
        *,
        prompt: str,
        duration_s: float,
        aspect_ratio: str,
        reference_image: bytes | None,
    ) -> tuple[bytes, float]:
        return self._videos.next(), STUB_COST_USD


def _is_last_line_of_scene(manifest: SceneManifest | None) -> list[bool]:
    """music_director.py calls TTSPort.speak() once per *dialogue line*, not
    once per scene -- a scene with N dialogue turns makes N calls, each
    returning one line's raw audio, which music_director then ffmpeg-concats
    into a single stored narration asset per scene (see
    _synthesize_scene_narration). Only that final concatenated clip ever
    gets persisted to storage, so a recording only has the *last* call's
    "true" output available per scene -- this precomputes, in call order,
    whether each expected speak() call is the last one for its scene."""
    if manifest is None:
        return []
    flags: list[bool] = []
    for scene in manifest.scenes:
        n = len(scene.dialogue) if scene.dialogue else 1
        flags.extend([False] * (n - 1) + [True])
    return flags


class RecordedTTS:
    """Replays the real recorded narration clip on each scene's last
    speak() call; earlier calls within a multi-line dialogue scene (whose
    individual raw audio was never persisted -- see
    _is_last_line_of_scene) get a cheap synthetic silent placeholder
    instead. music_director's ffmpeg concat then produces a scene clip
    close to (not byte-identical to) the original recording: the leading
    placeholder segments are near-silent and short, so they barely shift
    the real content that follows."""

    def __init__(self, fixtures: RecordedFixtures) -> None:
        self._narration = fixtures._narration
        self._is_last = iter(_is_last_line_of_scene(fixtures.manifest))

    async def speak(self, *, text: str, voice: str) -> tuple[bytes, float]:
        try:
            is_last = next(self._is_last)
        except StopIteration:
            is_last = True  # no manifest recorded -- assume 1 call per scene
        if is_last:
            return self._narration.next(), STUB_COST_USD
        return _stub_audio_bytes(), STUB_COST_USD


def _expected_scene_texts(manifest: SceneManifest | None) -> list[str]:
    if manifest is None:
        return []
    texts: list[str] = []
    for scene in manifest.scenes:
        texts.append(
            " ".join(turn.line for turn in scene.dialogue) if scene.dialogue else scene.narration
        )
    return texts


class RecordedTranscribe:
    """Always returns a transcript that exactly matches the corresponding
    scene's manifest text (dialogue joined, or narration), in scene order --
    guarantees Critic's audio-WER check passes regardless of the replayed
    audio's actual spoken content. Faithfully transcribing real recorded
    audio isn't the point of a recorded-replay fixture: the point is
    deterministic, cost-free realistic content for exercising *downstream*
    logic (Editor, UI, ...), not re-litigating Critic's own judgment against
    real audio every replay.
    Critic's `_score_audio` (one call per scene) is what this is sized for.
    Editor's captioning step (WP5) also transcribes once more, on the whole
    assembled final track, after Critic's per-scene calls are done -- that
    call gets every scene's text joined together rather than raising once
    the per-scene sequence is exhausted, since nothing downstream compares
    Editor's transcript against an "expected" value the way Critic does; it
    only needs *some* reasonable word-timed text to burn captions from.

    SPEC-GAP: only covers topical/dialogue-mode scenes (transcribed one
    narration asset per scene); a rhyme-mode recording (one AUDIO_MASTER
    transcribed against manifest.lyrics) isn't handled since no rhyme-mode
    fixture has been recorded yet.
    """

    def __init__(self, fixtures: RecordedFixtures) -> None:
        texts = _expected_scene_texts(fixtures.manifest)
        self._texts = _Sequence(texts, label="transcript")
        self._fallback_text = " ".join(texts)

    async def transcribe(self, *, audio: bytes) -> tuple[list[dict[str, str | float]], float]:
        try:
            text = self._texts.next()
        except IndexError:
            text = self._fallback_text
        words: list[dict[str, str | float]] = []
        start = 0.0
        for word in text.split():
            words.append({"word": word, "start": start, "end": start + 0.3})
            start += 0.3
        return words, STUB_COST_USD


class RecordedMusicGen:
    def __init__(self, fixtures: RecordedFixtures) -> None:
        self._audio_bed = fixtures._audio_bed

    async def generate(
        self,
        *,
        mode: Mode,
        lyrics: str | None,
        description: str,
        duration_s: float,
    ) -> tuple[bytes, float]:
        return self._audio_bed.next(), STUB_COST_USD


def recorded_port_bundle(
    root: Path | None = None, *, fixtures: RecordedFixtures | None = None
) -> PortBundle:
    """A PortBundle backed by real recorded content (see RecordedFixtures)
    instead of stubs.py's synthetic assets -- PublishPort/NotifyPort stay
    the plain stubs (nothing "recorded" about a no-op publish/notify).
    Storage is `fixtures.storage`, not a fresh StubStorage: it's already
    seeded with the show's real avatar bytes under the URIs `fixtures.style`
    (used to `db.save_show(...)` a matching show) actually points at.

    Pass an already-loaded `fixtures` (e.g. one also used to seed a show
    via `fixtures.style`) rather than `root` when the caller needs that same
    consistency -- constructing two separate RecordedFixtures against the
    same directory would give each its own StubStorage, and a style/storage
    pair built from different instances won't resolve against each other.
    """
    if fixtures is None:
        if root is None:
            raise ValueError("recorded_port_bundle needs either `root` or `fixtures`")
        fixtures = RecordedFixtures(root)
    return PortBundle(
        llm=RecordedLLM(fixtures),
        image_gen=RecordedImageGen(fixtures),
        video_gen=RecordedVideoGen(fixtures),
        music_gen=RecordedMusicGen(fixtures),
        tts=RecordedTTS(fixtures),
        transcribe=RecordedTranscribe(fixtures),
        storage=fixtures.storage,
        publish=StubPublish(),
        notify=StubNotify(),
    )
