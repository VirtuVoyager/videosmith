from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
from storysmith.agents import music_director
from storysmith.agents.director import _scene_violations, _significant_words
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


def test_regression_i2v_overlap_check_ignores_show_identity_vocabulary() -> None:
    """A real live run (show=Crocky & Roachy) hit this: a show's user-authored
    character description/personality can be long and detailed, and
    self-containment (§2.2) requires scene_image_prompt to restate it in
    full every time. Before this fix, a video_prompt that legitimately named
    which character was moving (reusing some of that same identity
    vocabulary, not re-describing layout) tripped the Amendment 01 §3
    composition-overlap check and crashed Director's post-validation even
    after the corrective round -- confirmed via `uv run python -c ...`
    reproducing the exact scene/style from that run before this fix landed."""
    style = StyleContract(
        art_style=(
            "Polished realistic 3D Disney-Pixar style CG animation, cinematic lighting, "
            "expressive stylized proportions, subsurface skin scattering"
        ),
        palette=[],
        mood="warm and witty",
        tempo_bpm=90,
        characters=[
            CharacterRef(
                name="Crocky",
                description=(
                    "a brown cockroach anthropomorphized to stand and sit upright like a "
                    "person, glossy chitin exoskeleton, large expressive eyes, small round "
                    "glasses, tweed vest over a collared shirt"
                ),
                personality="the intellectual worrier of the pair",
            ),
            CharacterRef(
                name="Roachy",
                description=(
                    "a reddish-brown cockroach anthropomorphized to stand and sit upright "
                    "like a person, glossy chitin exoskeleton, large expressive eyes, a "
                    "small bowtie"
                ),
                personality="the laid-back optimist",
            ),
        ],
        pacing_rules="",
        negative_terms=[],
    )
    scene = Scene(
        index=0,
        duration_s=6,
        gen_mode=SceneGenMode.I2V,
        scene_image_prompt=(
            "Polished realistic 3D Disney-Pixar style CG animation, cinematic lighting, "
            "expressive stylized proportions, subsurface skin scattering. A cozy cafe "
            "interior with warm lighting. Crocky, a brown cockroach anthropomorphized to "
            "stand and sit upright like a person, glossy chitin exoskeleton, large "
            "expressive eyes, small round glasses, tweed vest over a collared shirt, sits "
            "on the left holding a coffee cup. Roachy, a reddish-brown cockroach "
            "anthropomorphized to stand and sit upright like a person, glossy chitin "
            "exoskeleton, large expressive eyes, a small bowtie, sits on the right holding "
            "a coffee cup. Both characters are centered in frame, camera angle is a medium "
            "shot from slightly above, background shows cafe decor."
        ),
        video_prompt=(
            "Crocky and Roachy sip their coffee, occasionally gesturing with their hands as "
            "they talk, camera remains static, background stays fixed, nothing else moves."
        ),
        narration="",
    )
    style_words = [w.strip(",.") for w in style.art_style.lower().split() if len(w.strip(",.")) > 3]
    character_names = {c.name for c in style.characters}
    identity_words = _significant_words(style.art_style)
    for c in style.characters:
        identity_words |= _significant_words(c.name)
        identity_words |= _significant_words(c.description)
        identity_words |= _significant_words(c.personality)

    violations = _scene_violations(scene, style_words, character_names, identity_words)

    assert violations == []


def _crocky_and_roachy() -> StyleContract:
    return StyleContract(
        art_style="polished realistic 3D Disney-Pixar style CG animation",
        palette=[],
        mood="warm and witty",
        tempo_bpm=90,
        characters=[
            CharacterRef(
                name="Crocky",
                description=(
                    "a brown cockroach anthropomorphized to stand and sit upright like a "
                    "person, glossy chitin exoskeleton, large expressive eyes, small round "
                    "glasses, tweed vest over a collared shirt"
                ),
            ),
            CharacterRef(
                name="Roachy",
                description=(
                    "a reddish-brown cockroach anthropomorphized to stand and sit upright "
                    "like a person, glossy chitin exoskeleton, large expressive eyes, a "
                    "small bowtie"
                ),
            ),
        ],
        pacing_rules="",
        negative_terms=[],
    )


def test_regression_i2v_prompt_naming_character_without_description_is_flagged() -> None:
    """A real live run (show=Crocky & Roachy) hit this for real: only scene
    0's scene_image_prompt restated character appearance in any detail;
    scenes 1-4 just said "Crocky and Roachy sit at the table" -- self-
    containment (§2.2) was never actually checked per character, only that
    *some* art-style word appeared somewhere. The generated video showed
    five visually unrelated character designs, scene to scene, because nothing
    in the image model's stateless calls carried Crocky/Roachy's appearance
    forward. Before this fix, _validation_violations never caught this."""
    style = _crocky_and_roachy()
    scene = Scene(
        index=1,
        duration_s=6,
        gen_mode=SceneGenMode.I2V,
        scene_image_prompt=(
            "Same café view as previous scene, with Crocky and Roachy seated at the round "
            "table, coffee cups in front of them. Crocky on the left, Roachy on the right."
        ),
        video_prompt="Crocky and Roachy sip their coffee, camera remains static.",
        narration="",
    )
    style_words = [w.strip(",.") for w in style.art_style.lower().split() if len(w.strip(",.")) > 3]

    violations = _scene_violations(
        scene, style_words, {"Crocky", "Roachy"}, characters=style.characters
    )

    assert any("Crocky" in v and "doesn't restate" in v for v in violations)
    assert any("Roachy" in v and "doesn't restate" in v for v in violations)


def test_i2v_prompt_fully_restating_character_appearance_is_not_flagged() -> None:
    style = _crocky_and_roachy()
    scene = Scene(
        index=0,
        duration_s=6,
        gen_mode=SceneGenMode.I2V,
        scene_image_prompt=(
            "Polished realistic 3D Disney-Pixar style CG animation. A cozy cafe interior. "
            "Crocky, a brown cockroach anthropomorphized to stand and sit upright like a "
            "person, glossy chitin exoskeleton, large expressive eyes, small round glasses, "
            "tweed vest over a collared shirt, sits on the left holding a coffee cup. Roachy, "
            "a reddish-brown cockroach anthropomorphized to stand and sit upright like a "
            "person, glossy chitin exoskeleton, large expressive eyes, a small bowtie, sits "
            "on the right holding a coffee cup."
        ),
        video_prompt="Crocky and Roachy sip their coffee, camera remains static.",
        narration="",
    )
    style_words = [w.strip(",.") for w in style.art_style.lower().split() if len(w.strip(",.")) > 3]

    violations = _scene_violations(
        scene, style_words, {"Crocky", "Roachy"}, characters=style.characters
    )

    assert not any("doesn't restate" in v for v in violations)


def test_regression_deterministic_patch_fixes_missing_restatement_after_corrective_round() -> None:
    """Real live finding: even with the improved corrective-round wording
    above, a raw-completion model (Replicate) failed this exact check on
    3+ scenes two runs in a row -- spending a second paid LLM call and
    hoping isn't reliable enough. _patch_missing_character_restatements is
    the deterministic second line of defense: guarantees the
    self-containment property via string concatenation, no more API calls,
    no chance of crashing the run over it."""
    from storysmith.agents.director import (
        _patch_missing_character_restatements,
        _validation_violations,
    )

    style = _crocky_and_roachy()
    bad_scenes = [
        Scene(
            index=0,
            duration_s=6,
            gen_mode=SceneGenMode.I2V,
            scene_image_prompt=(
                "Polished realistic 3D Disney-Pixar style CG animation. Crocky, a brown "
                "cockroach anthropomorphized to stand and sit upright like a person, glossy "
                "chitin exoskeleton, large expressive eyes, small round glasses, tweed vest "
                "over a collared shirt, and Roachy, a reddish-brown cockroach anthropomorphized "
                "to stand and sit upright like a person, glossy chitin exoskeleton, large "
                "expressive eyes, a small bowtie, sit at a cafe table."
            ),
            video_prompt="Crocky and Roachy sip their coffee, camera remains static.",
            narration="",
        ),
    ] + [
        Scene(
            index=i,
            duration_s=6,
            gen_mode=SceneGenMode.I2V,
            # Exactly the real live failure: art style restated (so *that*
            # check passes, isolating the one this test targets), but
            # characters named with no restatement of their appearance.
            scene_image_prompt=(
                "Polished realistic 3D Disney-Pixar style CG animation. Same cafe view. "
                "Crocky and Roachy chat at the table."
            ),
            video_prompt="Crocky and Roachy sip their coffee, camera remains static.",
            narration="",
        )
        for i in range(1, 5)
    ]
    manifest = SceneManifest(
        title="t", description="d", tags=[], total_duration_s=30.0, music_cues=[], scenes=bad_scenes
    )
    assert _validation_violations(manifest, style) != []  # sanity: really is broken first

    patched = _patch_missing_character_restatements(manifest, style)

    assert _validation_violations(patched, style) == []
    # scene 0 (already fine) is untouched byte-for-byte
    assert patched.scenes[0] == manifest.scenes[0]


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
