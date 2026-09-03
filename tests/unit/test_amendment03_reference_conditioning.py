from __future__ import annotations

import io
from typing import Any

import pytest
from PIL import Image
from storysmith.agents import scene_stills
from storysmith.agents.scene_stills import _build_reference_image, _stitch_horizontally
from storysmith.models import (
    AssetKind,
    CharacterRef,
    Mode,
    Scene,
    SceneGenMode,
    SceneManifest,
    StyleContract,
    VideoProject,
)
from storysmith.pipeline import PortBundle
from storysmith.settings import Settings
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

pytestmark = pytest.mark.amendment03


def _png_bytes(*, width: int, height: int, color: str) -> bytes:
    image = Image.new("RGB", (width, height), color)
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def _style(*, one_character: bool = False) -> StyleContract:
    characters = [CharacterRef(name="Crocky", description="a brown cockroach")]
    if not one_character:
        characters.append(CharacterRef(name="Roachy", description="a reddish-brown cockroach"))
    return StyleContract(
        art_style="polished realistic 3D Disney-Pixar style CG animation",
        palette=[],
        mood="warm",
        tempo_bpm=90,
        characters=characters,
        pacing_rules="",
        negative_terms=[],
    )


def _manifest() -> SceneManifest:
    scene = Scene(
        index=0,
        duration_s=6,
        gen_mode=SceneGenMode.I2V,
        scene_image_prompt="A cozy cafe. Crocky, a brown cockroach, sits with a coffee cup.",
        video_prompt="Crocky sips coffee, camera remains static.",
        narration="",
    )
    return SceneManifest(
        title="t", description="d", tags=[], total_duration_s=6.0, music_cues=[], scenes=[scene]
    )


class _CapturingImageGen:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def generate(
        self, *, prompt: str, aspect_ratio: str, reference_image: bytes | None = None
    ) -> tuple[bytes, float]:
        self.calls.append(
            {"prompt": prompt, "aspect_ratio": aspect_ratio, "reference_image": reference_image}
        )
        return b"STILL", 0.04


def _ports(*, image_gen: Any = None, storage: Any = None) -> PortBundle:
    return PortBundle(
        llm=StubLLM(),
        image_gen=image_gen or StubImageGen(),
        video_gen=StubVideoGen(),
        music_gen=StubMusicGen(),
        tts=StubTTS(),
        transcribe=StubTranscribe(),
        storage=storage or StubStorage(),
        publish=StubPublish(),
        notify=StubNotify(),
    )


def test_stitch_horizontally_combines_two_differently_sized_images() -> None:
    a = _png_bytes(width=100, height=200, color="red")
    b = _png_bytes(width=300, height=100, color="blue")

    combined = _stitch_horizontally([a, b])

    result = Image.open(io.BytesIO(combined))
    # Both resized to the shorter image's height (100); widths scale to match.
    assert result.height == 100
    assert result.width == 50 + 300  # a's 100x200 -> 50x100 at target height 100


async def test_build_reference_image_returns_none_without_any_avatars() -> None:
    style = _style()  # no character has image_uri set

    result = await _build_reference_image(style, _ports())

    assert result is None


async def test_build_reference_image_returns_single_avatar_unstitched() -> None:
    storage = StubStorage()
    uri = await storage.put(key="crocky.png", data=b"SOLO_AVATAR", content_type="image/png")
    style = _style(one_character=True)
    style = style.model_copy(
        update={"characters": [style.characters[0].model_copy(update={"image_uri": uri})]}
    )

    result = await _build_reference_image(style, _ports(storage=storage))

    assert result == b"SOLO_AVATAR"  # returned as-is, no stitching for a single image


async def test_build_reference_image_stitches_two_avatars() -> None:
    storage = StubStorage()
    crocky_bytes = _png_bytes(width=100, height=100, color="brown")
    roachy_bytes = _png_bytes(width=100, height=100, color="orange")
    crocky_uri = await storage.put(key="crocky.png", data=crocky_bytes, content_type="image/png")
    roachy_uri = await storage.put(key="roachy.png", data=roachy_bytes, content_type="image/png")
    style = _style()
    style = style.model_copy(
        update={
            "characters": [
                style.characters[0].model_copy(update={"image_uri": crocky_uri}),
                style.characters[1].model_copy(update={"image_uri": roachy_uri}),
            ]
        }
    )

    result = await _build_reference_image(style, _ports(storage=storage))

    assert result is not None
    combined = Image.open(io.BytesIO(result))
    assert combined.width == 200  # two 100-wide crops, same height, side by side
    assert combined.height == 100


async def test_generate_one_passes_reference_image_and_uses_scene_image_model(
    settings_test: Settings,
) -> None:
    storage = StubStorage()
    avatar_bytes = _png_bytes(width=100, height=100, color="brown")
    avatar_uri = await storage.put(key="crocky.png", data=avatar_bytes, content_type="image/png")
    style = _style(one_character=True)
    style = style.model_copy(
        update={"characters": [style.characters[0].model_copy(update={"image_uri": avatar_uri})]}
    )
    state = VideoProject(
        project_id="a03-1", mode=Mode.RHYME, brief="b", style=style, manifest=_manifest()
    )
    image_gen = _CapturingImageGen()

    result = await scene_stills.run(
        state, ports=_ports(image_gen=image_gen, storage=storage), settings=settings_test
    )

    assert len(image_gen.calls) == 1
    assert image_gen.calls[0]["reference_image"] == avatar_bytes
    assert len(result["assets"]) == 1
    assert result["assets"][0].kind == AssetKind.SCENE_STILL


async def test_content_hash_differs_with_and_without_reference_image(
    settings_test: Settings,
) -> None:
    """Idempotency (§3.2) hashes on (model, prompt, reference image) -- a
    scene generated once with a frozen avatar and once without (e.g. before
    char_refs ran) must not be treated as the same request."""
    storage = StubStorage()
    avatar_bytes = _png_bytes(width=100, height=100, color="brown")
    avatar_uri = await storage.put(key="crocky.png", data=avatar_bytes, content_type="image/png")

    style_without = _style(one_character=True)
    state_without = VideoProject(
        project_id="a03-2", mode=Mode.RHYME, brief="b", style=style_without, manifest=_manifest()
    )
    style_with = style_without.model_copy(
        update={
            "characters": [style_without.characters[0].model_copy(update={"image_uri": avatar_uri})]
        }
    )
    state_with = VideoProject(
        project_id="a03-3", mode=Mode.RHYME, brief="b", style=style_with, manifest=_manifest()
    )

    result_without = await scene_stills.run(
        state_without, ports=_ports(storage=storage), settings=settings_test
    )
    result_with = await scene_stills.run(
        state_with, ports=_ports(storage=storage), settings=settings_test
    )

    assert result_without["assets"][0].content_hash != result_with["assets"][0].content_hash
