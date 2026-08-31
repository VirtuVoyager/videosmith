from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
from storysmith.agents import music_director
from storysmith.agents.director import _scene_violations
from storysmith.models import (
    AssetKind,
    CharacterRef,
    DialogueLine,
    Mode,
    Scene,
    SceneGenMode,
    SceneManifest,
    StyleContract,
    VideoProject,
)
from storysmith.pipeline import PortBundle
from storysmith.settings import Settings
from storysmith.util.ffmpeg import build_audio_concat_cmd, probe_duration
from storysmith_adapters.stubs import (
    StubImageGen,
    StubLLM,
    StubMusicGen,
    StubNotify,
    StubPublish,
    StubStorage,
    StubTranscribe,
    StubTTS,
    StubVideoGen,
)

pytestmark = pytest.mark.amendment02


def _t2v_scene(*, index: int, dialogue: list[DialogueLine] | None) -> Scene:
    return Scene(
        index=index,
        duration_s=8,
        video_prompt="soft 2D cutout animation duck",
        narration="",
        gen_mode=SceneGenMode.T2V,
        dialogue=dialogue,
    )


def test_dialogue_speaker_must_match_a_cast_member() -> None:
    scene = _t2v_scene(index=0, dialogue=[DialogueLine(speaker="Ghost", line="boo")])
    violations = _scene_violations(scene, style_words=[], character_names={"Bob", "Miko"})
    assert any("Ghost" in v for v in violations)


def test_dialogue_with_known_speakers_has_no_speaker_violation() -> None:
    scene = _t2v_scene(
        index=0,
        dialogue=[
            DialogueLine(speaker="Bob", line="hey"),
            DialogueLine(speaker="Miko", line="hi!"),
        ],
    )
    violations = _scene_violations(scene, style_words=[], character_names={"Bob", "Miko"})
    assert violations == []


def test_scene_without_dialogue_is_unaffected() -> None:
    scene = _t2v_scene(index=0, dialogue=None)
    violations = _scene_violations(scene, style_words=[], character_names={"Bob"})
    assert violations == []


def test_build_audio_concat_cmd_single_clip() -> None:
    cmd = build_audio_concat_cmd([Path("a.mp3")], [1.0], 0.2, Path("out.wav"))
    filter_complex = cmd[cmd.index("-filter_complex") + 1]
    assert "[0:a]adelay=0|0[c0]" in filter_complex
    assert "[c0]anull[out]" in filter_complex
    assert cmd[cmd.index("-map") + 1] == "[out]"


def test_build_audio_concat_cmd_multiple_clips_sequenced_with_gap() -> None:
    cmd = build_audio_concat_cmd([Path("a.mp3"), Path("b.mp3")], [1.0, 2.0], 0.5, Path("out.wav"))
    filter_complex = cmd[cmd.index("-filter_complex") + 1]
    assert "[0:a]adelay=0|0[c0]" in filter_complex
    assert "[1:a]adelay=1500|1500[c1]" in filter_complex  # 1.0s clip + 0.5s gap
    assert "[c0][c1]amix=inputs=2:normalize=0[out]" in filter_complex


def test_build_audio_concat_cmd_requires_at_least_one_clip() -> None:
    with pytest.raises(AssertionError):
        build_audio_concat_cmd([], [], 0.2, Path("out.wav"))


def _ports() -> PortBundle:
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


async def test_music_director_dialogue_produces_one_narration_asset_per_scene(
    settings_test: Settings,
) -> None:
    if settings_test.skip_ffmpeg:
        pytest.skip("SS_SKIP_FFMPEG=1")

    style = StyleContract(
        art_style="soft 2D cutout animation",
        palette=["#FFFFFF"],
        mood="cheerful",
        tempo_bpm=100,
        characters=[
            CharacterRef(name="Bob", description="a cat", voice_id="am_adam"),
            CharacterRef(name="Miko", description="a dog", voice_id="af_bella"),
        ],
        pacing_rules="",
        negative_terms=[],
    )
    scene = Scene(
        index=0,
        duration_s=8,
        video_prompt="p",
        narration="",
        gen_mode=SceneGenMode.T2V,
        dialogue=[
            DialogueLine(speaker="Bob", line="Are you going to eat that?"),
            DialogueLine(speaker="Miko", line="Every last bite!"),
        ],
    )
    manifest = SceneManifest(
        title="t",
        description="d",
        tags=[],
        total_duration_s=8,
        lyrics=None,
        music_cues=[],
        scenes=[scene],
    )
    state = VideoProject(
        project_id="amendment02-dialogue",
        mode=Mode.TOPICAL,
        brief="b",
        style=style,
        manifest=manifest,
    )
    ports = _ports()

    result = await music_director.run(state, ports=ports, settings=settings_test)

    narration_assets = [
        a
        for a in result["assets"]
        if a.kind == AssetKind.AUDIO_MASTER and a.meta.get("role") == "narration"
    ]
    assert len(narration_assets) == 1  # not one per dialogue line
    assert narration_assets[0].scene_index == 0

    audio_bytes = await ports.storage.get(uri=narration_assets[0].uri)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "combined.wav"
        path.write_bytes(audio_bytes)
        duration = probe_duration(path)
    # two 1s stub clips + a 0.2s gap between them, roughly
    assert duration == pytest.approx(2.2, abs=0.3)
