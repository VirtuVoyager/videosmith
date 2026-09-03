# StorySmith — Amendment 04: QA-Stage Provider-Error Resilience

Status: AMENDMENT. Layers on top of `HANDOFF_SPEC.md` (WP1-8) and Amendments
01-03, all unchanged and remaining the base contract. This document describes
only the diff. Touches WP1 (`models.py`, `QAVerdict`), WP5 (`editor.py`), WP6
(`critic.py`), and WP7 (`graph/nodes.py::review_gate`). Does not touch
Director, Videographer, Music Director, or the graph topology.

## 1. Problem observed (evidence, not theory)

A real live episode ran every scene through generation successfully (real
money already spent on video, stills, and audio), then Critic's QA stage hit
Replicate's account-level low-credit throttle
(`HTTP 429: "reduced to 6 requests per minute ... while you have less than
$5.0 in credit"`) mid-run. `critic.py` had no handling for a QA call itself
failing -- the exception propagated up uncaught and crashed the whole
pipeline. Result: every already-generated, already-paid-for asset was
discarded with no `FINAL_VIDEO` produced -- a total loss for zero
deliverable, purely because the *judging* step (not the content) hit an
infrastructure problem.

## 2. Fix: a QA-stage failure is not a content verdict

The existing `QAVerdict` enum (`PASS`, `RETRY`, `HUMAN_REVIEW`) has no value
for "this was never actually judged." Overloading `HUMAN_REVIEW` for a
provider outage would work operationally (both route to `review_gate`) but
would be dishonest: `HUMAN_REVIEW` means a real content concern was found,
and mixing "the model said no" with "the model never got to answer" makes
`review_gate`'s escalation summary misleading, and permanently discards
whichever scenes the outage happened to hit.

New value: `QAVerdict.INCONCLUSIVE` — QA could not run for this
scene/audio track due to a provider error. It is deliberately routed like
`PASS`, not like `HUMAN_REVIEW` or `RETRY`: there is no content reason to
withhold or regenerate a scene nobody actually looked at, so the run should
still reach `editor` and produce a deliverable, with the human reviewer told
exactly which parts were never really checked.

## 3. Changes

### `models.py`
`QAVerdict` gains `INCONCLUSIVE = "inconclusive"`.

### `critic.py`
Both QA call sites in `run()` — the per-scene `_score_scene` call and the
`_score_audio` call — are wrapped in `try/except Exception`. On failure, a
`QAReport` is appended with `verdict=QAVerdict.INCONCLUSIVE`, empty scores,
no safety flags, and a critique naming the provider error, and the loop
continues (or, for audio, the `else` branch of the same try/except is simply
skipped) rather than propagating. The two failure sites are independent —
a scene-QA outage doesn't block audio QA from still running normally, and
vice versa.

### `graph/build.py::_critic_router`
**No change needed.** The router already defaults to `return destinations or
["editor"]` when nothing has `HUMAN_REVIEW` and no `RETRY` destinations were
added — `INCONCLUSIVE` matches neither condition, so it falls through to
`editor` exactly like an all-`PASS` state would.

### `editor.py::_latest_passing_scene_videos`
This filter selected scenes by `verdict == QAVerdict.PASS` only. Without a
change here, an `INCONCLUSIVE` scene would have been silently dropped from
the assembled final video even though the router sent the project to
`editor` as if everything passed — defeating the entire point of the new
verdict. Fixed: the filter now matches `verdict in (QAVerdict.PASS,
QAVerdict.INCONCLUSIVE)`. `RETRY` and `HUMAN_REVIEW` scenes are still
excluded, unchanged.

### `graph/nodes.py::review_gate`
The Telegram notification only ever surfaced `HUMAN_REVIEW` escalations. It
now also collects `INCONCLUSIVE` reports into a second, separate line
(`"NOT actually QA-checked (provider error) -- verify manually: ..."`) so a
human approving the run knows which scenes/audio in the assembled video were
never actually judged, without conflating that with a real content
escalation in the headline.

### UI (`apps/ui/app/p/[id]/page.tsx`)
`VerdictBadge` gains a distinct gray style for `"inconclusive"`, separate
from the green `"pass"`, amber `"retry"`, and red (`"human_review"` and any
other value) styles — so it reads as "not checked" rather than "flagged."

## 4. What was deliberately left unchanged

- `videographer.py`/`scene_stills.py`'s retry-selection filters match on
  `QAVerdict.RETRY` specifically, so `INCONCLUSIVE` correctly never triggers
  a regeneration (there's nothing wrong with the asset to fix).
- Audio assembly in `editor.py` was already unconditional on the audio
  `QAReport`'s verdict (it just uses whichever `AUDIO_MASTER`/narration
  assets exist), so an `INCONCLUSIVE` audio verdict needed no corresponding
  change there.

## 5. Cost

None. This is pure failure-handling — no new paid API calls are introduced.
The actual saving is avoiding a second full paid regeneration pass after a
crash: previously, a QA-stage outage meant re-running the entire episode
from scratch (script, stills, video, music all regenerated) to get anything
at all.

## 6. Test additions

- `tests/unit/test_wp6_critic.py`: `test_regression_scene_qa_provider_error_yields_inconclusive_not_crash`,
  `test_regression_audio_qa_provider_error_yields_inconclusive_not_crash`.
- `tests/unit/test_wp6_graph_router.py`: `test_router_inconclusive_goes_to_editor_like_pass`,
  `test_router_inconclusive_scene_with_passing_audio_goes_to_editor`.
- `tests/unit/test_amendment04_qa_resilience.py` (mark `amendment04`):
  `editor._latest_passing_scene_videos` includes `INCONCLUSIVE` alongside
  `PASS` and still excludes `RETRY`/`HUMAN_REVIEW`; `review_gate`'s
  notification text surfaces an inconclusive scene distinctly from a real
  escalation.

## 7. Migration note for the coding agent

Additive throughout — no existing checkpoints, tests, or call sites break.
`INCONCLUSIVE` is a new enum member; any code pattern-matching on the closed
set `{PASS, RETRY, HUMAN_REVIEW}` was audited (`grep -rn "QAVerdict\."`) and
only `editor.py`'s scene filter needed updating to also treat it as
"include in the cut."
