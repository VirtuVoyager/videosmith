from __future__ import annotations

import pytest
from storysmith.agents.editor import _latest_passing_scene_videos
from storysmith.graph.nodes import review_gate
from storysmith.models import (
    AssetKind,
    AssetRef,
    Mode,
    QAReport,
    QAVerdict,
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

pytestmark = pytest.mark.amendment04

# Confirmed live: a real run ran out of Replicate credit mid-QA. The
# unhandled exception crashed the whole pipeline even though every scene had
# already been generated (real money already spent) -- total loss, no
# final_video, nothing to show for it. Amendment 04 introduces
# QAVerdict.INCONCLUSIVE for exactly this case (see its docstring in
# models.py): a QA-stage provider outage is not a content problem, so it
# must not behave like HUMAN_REVIEW or RETRY -- it should still let the
# already-paid-for assets reach a final assembled video. critic.py's own
# try/except -> INCONCLUSIVE behavior is covered by
# tests/unit/test_wp6_critic.py's two new regression tests, and the router
# fallthrough by tests/unit/test_wp6_graph_router.py's two new tests. This
# file covers the two remaining places that needed to actually treat
# INCONCLUSIVE like PASS rather than silently dropping or misreporting it.


def _report(scene_index: int | None, verdict: QAVerdict) -> QAReport:
    return QAReport(
        scene_index=scene_index, verdict=verdict, scores={}, safety_flags=[], critique=""
    )


def _scene_asset(index: int, attempt: int = 1) -> AssetRef:
    return AssetRef(
        kind=AssetKind.SCENE_VIDEO,
        scene_index=index,
        attempt=attempt,
        uri=f"file:///scene_{index}.mp4",
        content_hash=f"h{index}",
    )


def test_editor_includes_inconclusive_scenes_in_final_cut() -> None:
    """Without this, editor.py's _latest_passing_scene_videos filtered on
    QAVerdict.PASS alone -- an INCONCLUSIVE scene (QA never actually ran)
    would have silently vanished from the assembled video even though the
    router sends the whole project to editor as if it passed. RETRY and
    HUMAN_REVIEW scenes must still be excluded -- only a real content
    escalation or a pending retry should hold a scene back."""
    state = VideoProject(
        project_id="p",
        mode=Mode.RHYME,
        brief="b",
        assets=[_scene_asset(0), _scene_asset(1), _scene_asset(2), _scene_asset(3)],
        qa_reports=[
            _report(0, QAVerdict.PASS),
            _report(1, QAVerdict.INCONCLUSIVE),
            _report(2, QAVerdict.RETRY),
            _report(3, QAVerdict.HUMAN_REVIEW),
        ],
    )

    included = dict(_latest_passing_scene_videos(state))

    assert set(included) == {0, 1}


async def test_review_gate_flags_inconclusive_scenes_separately_from_escalations(
    settings_test: Settings,
) -> None:
    """A human approving the run needs to know scene 1 was never actually
    QA-checked (provider error), distinct from a real HUMAN_REVIEW content
    escalation -- otherwise an unchecked scene could ship silently."""
    state = VideoProject(
        project_id="p",
        mode=Mode.RHYME,
        brief="counting ducks",
        qa_reports=[
            _report(0, QAVerdict.PASS),
            _report(1, QAVerdict.INCONCLUSIVE),
        ],
    )
    notify = StubNotify()
    ports = PortBundle(
        llm=StubLLM(),
        image_gen=StubImageGen(),
        video_gen=StubVideoGen(),
        music_gen=StubMusicGen(),
        tts=StubTTS(),
        transcribe=StubTranscribe(),
        storage=StubStorage(),
        publish=StubPublish(),
        notify=notify,
    )

    await review_gate(state, ports=ports, settings=settings_test)

    assert len(notify.sent) == 1
    text, _link = notify.sent[0]
    assert "needs human review" not in text  # not a real content escalation
    assert "scene 1" in text
    assert "NOT actually QA-checked" in text
