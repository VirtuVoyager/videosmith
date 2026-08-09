from __future__ import annotations

import pytest
from storysmith.agents.critic import _rubric_text, _weighted_score
from storysmith.settings import Settings
from storysmith.util.configs import load_rubric

pytestmark = pytest.mark.wp6


def test_load_rubric(settings_test: Settings) -> None:
    rubric = load_rubric(settings_test.configs_dir)
    names = {c.name for c in rubric.criteria}
    assert names == {
        "style_adherence",
        "character_consistency",
        "visual_artifacts",
        "kid_appeal",
        "safety",
        "lesson_clarity",
    }
    assert rubric.pass_threshold == pytest.approx(0.7)
    assert rubric.max_attempts_before_human_review == 3
    assert rubric.audio_max_attempts_before_human_review == 2


def test_weighted_score_all_max(settings_test: Settings) -> None:
    rubric = load_rubric(settings_test.configs_dir)
    scores = {c.name: 1.0 for c in rubric.criteria if not c.conditional}
    assert _weighted_score(rubric, scores, has_lesson=False) == pytest.approx(1.0)


def test_weighted_score_excludes_conditional_criteria_without_lesson(
    settings_test: Settings,
) -> None:
    rubric = load_rubric(settings_test.configs_dir)
    scores = {c.name: 1.0 for c in rubric.criteria}
    scores["lesson_clarity"] = 0.0  # would drag the average down if it counted
    weighted = _weighted_score(rubric, scores, has_lesson=False)
    assert weighted == pytest.approx(1.0)


def test_weighted_score_includes_conditional_criteria_with_lesson(settings_test: Settings) -> None:
    rubric = load_rubric(settings_test.configs_dir)
    scores = {c.name: 1.0 for c in rubric.criteria}
    scores["lesson_clarity"] = 0.0
    weighted = _weighted_score(rubric, scores, has_lesson=True)
    assert weighted < 1.0


def test_weighted_score_missing_criterion_scores_zero(settings_test: Settings) -> None:
    rubric = load_rubric(settings_test.configs_dir)
    assert _weighted_score(rubric, {}, has_lesson=False) == pytest.approx(0.0)


def test_rubric_text_omits_lesson_clarity_without_lesson(settings_test: Settings) -> None:
    rubric = load_rubric(settings_test.configs_dir)
    text = _rubric_text(rubric, has_lesson=False)
    assert "lesson_clarity" not in text
    assert "style_adherence" in text


def test_rubric_text_includes_lesson_clarity_with_lesson(settings_test: Settings) -> None:
    rubric = load_rubric(settings_test.configs_dir)
    text = _rubric_text(rubric, has_lesson=True)
    assert "lesson_clarity" in text
