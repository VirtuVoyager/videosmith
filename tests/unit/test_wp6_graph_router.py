from __future__ import annotations

import pytest
from storysmith.graph.build import _critic_router
from storysmith.models import Mode, QAReport, QAVerdict, VideoProject

pytestmark = pytest.mark.wp6


def _report(scene_index: int | None, verdict: QAVerdict) -> QAReport:
    return QAReport(
        scene_index=scene_index, verdict=verdict, scores={}, safety_flags=[], critique=""
    )


def _state(reports: list[QAReport]) -> VideoProject:
    return VideoProject(project_id="p", mode=Mode.RHYME, brief="b", qa_reports=reports)


def test_router_all_pass_goes_to_editor() -> None:
    state = _state([_report(0, QAVerdict.PASS), _report(None, QAVerdict.PASS)])
    assert _critic_router(state) == ["editor"]


def test_router_scene_retry_goes_to_videographer() -> None:
    state = _state([_report(0, QAVerdict.RETRY), _report(None, QAVerdict.PASS)])
    assert _critic_router(state) == ["videographer"]


def test_router_audio_retry_goes_to_music_director() -> None:
    state = _state([_report(0, QAVerdict.PASS), _report(None, QAVerdict.RETRY)])
    assert _critic_router(state) == ["music_director"]


def test_router_scene_and_audio_retry_fans_out_to_both() -> None:
    state = _state([_report(0, QAVerdict.RETRY), _report(None, QAVerdict.RETRY)])
    assert set(_critic_router(state)) == {"videographer", "music_director"}


def test_router_human_review_takes_priority_over_retry() -> None:
    state = _state([_report(0, QAVerdict.RETRY), _report(None, QAVerdict.HUMAN_REVIEW)])
    assert _critic_router(state) == ["review_gate"]


def test_router_scene_human_review_takes_priority_too() -> None:
    state = _state([_report(0, QAVerdict.HUMAN_REVIEW), _report(1, QAVerdict.PASS)])
    assert _critic_router(state) == ["review_gate"]
