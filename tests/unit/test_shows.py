from __future__ import annotations

from typing import Any

import pytest
from pydantic import BaseModel
from storysmith import db
from storysmith.agents import creative_director
from storysmith.models import CharacterRef, Mode, ProjectStatus, StyleContract, VideoProject
from storysmith.pipeline import Pipeline, PortBundle
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

pytestmark = pytest.mark.wp7


class _CountingLLM(StubLLM):
    def __init__(self) -> None:
        self.calls = 0

    async def complete_structured(self, **kwargs: Any) -> tuple[BaseModel, float]:
        self.calls += 1
        return await super().complete_structured(**kwargs)


class _CountingImageGen(StubImageGen):
    def __init__(self) -> None:
        self.calls = 0

    async def generate(self, **kwargs: Any) -> tuple[bytes, float]:
        self.calls += 1
        return await super().generate(**kwargs)


def _style(*, with_avatars: bool) -> StyleContract:
    return StyleContract(
        art_style="soft 2D cutout animation",
        palette=["#FFAA00"],
        mood="cheerful",
        tempo_bpm=100,
        characters=[
            CharacterRef(
                name="Bob",
                description="an orange cat",
                personality="sarcastic and lazy",
                voice_id="am_adam",
                image_uri="local://shows/demo/char_Bob.png" if with_avatars else None,
            ),
            CharacterRef(
                name="Miko",
                description="a golden retriever",
                personality="earnest and easily excited",
                voice_id="af_bella",
                image_uri="local://shows/demo/char_Miko.png" if with_avatars else None,
            ),
        ],
        pacing_rules="",
        negative_terms=[],
    )


async def test_save_and_load_show_round_trips(pg_required: Settings) -> None:
    style = _style(with_avatars=True)
    await db.save_show(pg_required.db_url, show_id="bob-and-miko", name="Bob & Miko", style=style)

    loaded = await db.load_show(pg_required.db_url, show_id="bob-and-miko")

    assert loaded is not None
    assert loaded.name == "Bob & Miko"
    round_tripped = StyleContract.model_validate_json(loaded.style_json)
    assert round_tripped == style


async def test_load_show_missing_returns_none(pg_required: Settings) -> None:
    assert await db.load_show(pg_required.db_url, show_id="does-not-exist") is None


async def test_save_show_upserts(pg_required: Settings) -> None:
    style_v1 = _style(with_avatars=True)
    await db.save_show(pg_required.db_url, show_id="upsert-show", name="v1", style=style_v1)
    style_v2 = style_v1.model_copy(update={"mood": "spooky"})
    await db.save_show(pg_required.db_url, show_id="upsert-show", name="v2", style=style_v2)

    loaded = await db.load_show(pg_required.db_url, show_id="upsert-show")
    assert loaded is not None
    assert loaded.name == "v2"
    assert StyleContract.model_validate_json(loaded.style_json).mood == "spooky"


async def test_list_shows_orders_newest_first(pg_required: Settings) -> None:
    await db.save_show(
        pg_required.db_url, show_id="show-a", name="A", style=_style(with_avatars=True)
    )
    await db.save_show(
        pg_required.db_url, show_id="show-b", name="B", style=_style(with_avatars=True)
    )

    rows = await db.list_shows(pg_required.db_url)
    ids = [r.show_id for r in rows]
    assert "show-a" in ids
    assert "show-b" in ids
    assert ids.index("show-b") < ids.index("show-a")  # b saved more recently


def _ports(llm: Any = None, image_gen: Any = None, storage: Any = None) -> PortBundle:
    return PortBundle(
        llm=llm or StubLLM(),
        image_gen=image_gen or StubImageGen(),
        video_gen=StubVideoGen(),
        music_gen=StubMusicGen(),
        tts=StubTTS(),
        transcribe=StubTranscribe(),
        storage=storage or StubStorage(),
        publish=StubPublish(),
        notify=StubNotify(),
    )


async def test_creative_director_skips_when_style_already_set(settings_test: Settings) -> None:
    llm = _CountingLLM()
    state = VideoProject(
        project_id="p1",
        mode=Mode.TOPICAL,
        brief="a topic",
        style=_style(with_avatars=True),
        show_id="bob-and-miko",
    )

    result = await creative_director.run(state, ports=_ports(llm=llm), settings=settings_test)

    assert result == {}
    assert llm.calls == 0


async def test_char_refs_skips_when_avatars_already_frozen(settings_test: Settings) -> None:
    from storysmith.graph.nodes import char_refs

    image_gen = _CountingImageGen()
    state = VideoProject(
        project_id="p1",
        mode=Mode.TOPICAL,
        brief="a topic",
        style=_style(with_avatars=True),
        show_id="bob-and-miko",
    )

    result = await char_refs(state, ports=_ports(image_gen=image_gen), settings=settings_test)

    assert result == {}
    assert image_gen.calls == 0


async def test_pipeline_run_with_show_id_loads_frozen_cast(pg_required: Settings) -> None:
    # Unlike the round-trip tests above, this runs the *full* graph, which
    # means critic.py really fetches each character's image_uri from storage
    # (for the vision-QA reference image) -- fake, never-stored URI strings
    # would 404/KeyError there. Seed real bytes into the same StubStorage the
    # pipeline uses, matching test_wp6_critic.py's _seeded_state pattern.
    storage = StubStorage()
    bob_uri = await storage.put(
        key="shows/frozen-cast/char_Bob.png", data=b"BOB", content_type="image/png"
    )
    miko_uri = await storage.put(
        key="shows/frozen-cast/char_Miko.png", data=b"MIKO", content_type="image/png"
    )
    style = _style(with_avatars=True).model_copy(
        update={
            "characters": [
                CharacterRef(name="Bob", description="an orange cat", image_uri=bob_uri),
                CharacterRef(name="Miko", description="a golden retriever", image_uri=miko_uri),
            ]
        }
    )
    await db.save_show(pg_required.db_url, show_id="frozen-cast", name="Frozen Cast", style=style)

    pipeline = Pipeline(settings=pg_required, ports=_ports(storage=storage))
    result = await pipeline.run(
        brief="a topic for this episode",
        mode=Mode.TOPICAL,
        project_id="show-episode-1",
        show_id="frozen-cast",
    )

    assert result.show_id == "frozen-cast"
    assert result.style is not None
    assert {c.name for c in result.style.characters} == {"Bob", "Miko"}
    assert result.status == ProjectStatus.REVIEW


async def test_pipeline_run_with_unknown_show_id_raises(pg_required: Settings) -> None:
    pipeline = Pipeline(settings=pg_required, ports=_ports())
    with pytest.raises(ValueError, match="no show found"):
        await pipeline.run(brief="x", mode=Mode.TOPICAL, project_id="p2", show_id="nope")


async def test_pipeline_run_with_show_id_requires_db_url(settings_test: Settings) -> None:
    pipeline = Pipeline(settings=settings_test, ports=_ports())
    with pytest.raises(ValueError, match="SS_DB_URL"):
        await pipeline.run(brief="x", mode=Mode.TOPICAL, project_id="p3", show_id="anything")
