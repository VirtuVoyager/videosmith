# StorySmith — Amendment 02: User-Authored Character Library & Multi-Character Dialogue

Status: AMENDMENT. Layers on top of `HANDOFF_SPEC.md` (WP1-8) and Amendment 01,
both unchanged and remaining the base contract. This document describes only the
diff. Touches WP1 (models, `db.py`), WP2 (Creative Director, Director,
Character Refs), WP4 (Music Director), and WP7 (API, UI). Does not touch WP3
(Videographer), WP5 (Editor), WP6 (Critic routing logic), or WP8 — a show
episode is, once its `StyleContract` is loaded, an ordinary pipeline run.

## 1. Problem observed

Sitcom-style shorts (a recurring, named cast that stays visually and
behaviorally consistent episode to episode) don't fit the base spec's model:
Creative Director invents a fresh 1-2 character cast from the brief on every
single run. There is no way to say "these are my two characters, always,"
and no way to represent two characters trading lines instead of one narration
track.

Deliberately **not** solved by having an LLM generate-then-persist a cast —
a cast worth building a show around is an authorial choice, not something to
delegate to a model. The user describes each character directly (appearance,
personality, quirks, voice); the system's job is to freeze that description
into a reusable `StyleContract` once, and reuse it verbatim thereafter.

## 2. Fix: shows as a frozen, user-authored `StyleContract`

A **show** is a `StyleContract` with every `CharacterRef.image_uri` already
populated, saved once under a `show_id`, created through a dedicated flow that
never calls an LLM. Triggering an episode with a `show_id` loads that
`StyleContract` straight into `VideoProject.style` and skips Creative Director
and Character Refs entirely (both become no-op guards, not new graph
topology) — everything downstream (Director onward) runs unmodified against
the loaded style.

## 3. Model changes (`storysmith-core/src/storysmith/models.py`)

Additive only — existing checkpoints remain loadable.

```python
class CharacterRef(BaseModel):
    name: str
    description: str             # visual description -- feeds image-gen prompts only
    image_uri: str | None = None
    personality: str = ""        # behavioral/voice guidance for the Director; kept
                                  # separate from `description` since it must never
                                  # leak into an image-generation prompt
    voice_id: str | None = None  # TTS voice for this character's dialogue lines

class DialogueLine(BaseModel):
    speaker: str                 # must exactly match a CharacterRef.name
    line: str

class Scene(BaseModel):
    # ... existing fields unchanged ...
    dialogue: list[DialogueLine] | None = None
    # None = today's single-narration behavior, unchanged. When set, `narration`
    # may be empty -- Music Director synthesizes `dialogue` instead (§6).

class VideoProject(BaseModel):
    # ... existing fields unchanged ...
    show_id: str | None = None
```

`graph/build.py`'s `_CHECKPOINT_ALLOWED_TYPES` gains `DialogueLine`.

## 4. Persistent shows (`db.py`, new — no LLM involved)

New table `shows` (`ShowRow`: `show_id` string PK, `name`, `style_json` text,
`created_at`), same `Base`/`Mapped` pattern as `CostEntryRow`/`ProjectRow`.
Migration `0003_create_shows.py`. `projects` gains a nullable `show_id` column
(migration `0004_add_show_id_to_projects.py`) so a project can record which
show it belongs to.

`db.py` additions: `save_show` (upsert), `load_show`, `list_shows`.
`upsert_project_snapshot` gains a `show_id` parameter.

The character-reference-portrait prompt (base spec §2.3) is extracted into
`storysmith.util.character_prompts.build_char_ref_prompt` so both the
existing `char_refs` node and the new show-creation endpoint (§5) build
portraits from one prompt shape, never two copies that can drift apart.

### 4.1 Character reference sheets, not single portraits

`build_char_ref_prompt` requests a full character-turnaround **model sheet**,
not one static pose: front / three-quarter / side / back full-body views in a
row, plus a small expression-study grid (neutral, talking, thoughtful, one
more), on a plain background, consistent proportions and color across every
view. Rationale: this reference image is what Critic's vision-QA compares
every scene keyframe against for the `character_consistency` rubric criterion
(base spec §6) — more views and expressions in one image gives it (and the
human reviewing the avatar gallery) far more surface to check consistency
against than a single neutral-pose portrait can.

The portrait is generated at a fixed `CHAR_REF_ASPECT_RATIO = "3:2"`
(landscape) regardless of the show's own `StyleContract.aspect_ratio`
(9:16) — this image is never composited into the final video, only shown in
the UI gallery and handed to Critic, so it isn't bound to the video's frame
shape, and a turnaround-sheet layout needs the width.

SPEC-GAP: `flux-schnell` (the current `image_model`) cannot reliably render
legible on-image text, so the prompt deliberately asks only for the *visual*
model-sheet structure (views + expressions), not text labels or hex-code
swatches the way a hand-drawn reference sheet would carry.

## 5. API additions (`apps/api/src/api/main.py`, bearer-protected)

- `POST /shows` — body: `show_id`, `name`, `art_style`, optional style fields
  (`palette`, `mood`, `tempo_bpm`, `aspect_ratio`, `resolution`,
  `pacing_rules`), `characters: [{name, description, personality, voice_id}]`.
  Builds a `StyleContract` directly from the input (safety negative-terms
  merged the same way `creative_director.py` does), generates one reference
  sheet per character concurrently via `ImageGenPort` + `build_char_ref_prompt`,
  uploads via `StoragePort`, records cost under `project_id=f"show:{show_id}"`
  (so it still flows into the daily budget cap), saves via `save_show`,
  returns the show with each character's `image_uri`. Rejects an empty cast
  with 422.
- `GET /shows` — list, for the UI's show picker.
- `GET /assets/view?uri=...` — streams any asset's bytes back through the
  server via the same `StoragePort.get()` every agent already uses, with an
  inferred content type. Needed because neither an `<img src>` nor a plain
  `<a href>` can carry a bearer `Authorization` header, and `local://` URIs
  aren't browser-fetchable at all — the client does `fetch()` + `Blob` +
  `URL.createObjectURL()` instead. Used for both the avatar gallery and the
  final-video download button (§7).
- `RunRequest` gains `show_id: str | None = None`; `trigger_run` 404s
  synchronously if the given `show_id` doesn't exist, before queuing the run.

## 6. Pipeline / graph wiring for a fixed cast

`Pipeline.run()` gains `show_id: str | None = None`. On a fresh run with
`show_id` set: `load_show`; missing show → `ValueError` (the API turns this
into the 404 above) rather than silently falling back to auto-generation.
`db_url` empty → `ValueError` (shows are Postgres-only). On success, construct
`VideoProject` with `style=<loaded>, status=ProjectStatus.STYLED, show_id=show_id`.

- `agents/creative_director.py`: `state.style is not None` on entry → return
  `{}` immediately (no LLM call, no cost).
- `graph/nodes.py::char_refs`: every character already has `image_uri` set →
  return `{}` immediately (no image-gen calls).

`apps/worker/src/worker/main.py` gains a `--show-id` CLI option threaded into
`Pipeline.run`.

## 7. Multi-character dialogue

`prompts/director.md` gains a "Multi-character dialogue" section: when the
`StyleContract` has 2+ characters, a scene *may* set `dialogue` (a list of
`{speaker, line}`, using each character's `personality` for voice/tone)
instead of `narration`; every `speaker` must exactly match a character name;
short exchanges, comedic timing.

`director.py`'s `_scene_violations` gains one check: every `dialogue[].speaker`
matches a `style.characters[].name` — feeds the existing corrective-round
mechanism (base spec §2.2's post-validation LLM round).

`music_director.py::_run_topical`: a scene with `dialogue` set has each line
synthesized separately via `TTSPort.speak(voice=<speaker's voice_id, falling
back to settings.tts_voice>)`, then concatenated (new
`util/ffmpeg.py::build_audio_concat_cmd`, `adelay`+`amix`, same pure-function/
golden-test style as the Editor's existing command builders) into **one**
combined clip stored as the scene's single narration `AssetRef` — identical
shape to today's single-voice path, which stays the fallback when `dialogue`
is `None`. This is fully contained within `music_director.py`: `editor.py`
and `util/assets.py::latest_narration_assets` (both hard-keyed to exactly one
narration asset per `scene_index`) need zero changes.

## 8. UI (`apps/ui`)

- `apps/ui/app/shows/new/page.tsx` — "Create a show": show id/name/art-style
  fields (art-style default: `"polished realistic 3D Disney-Pixar style CG
  animation, cinematic lighting, expressive stylized proportions, subsurface
  skin scattering"` — the currently-popular look for this format, distinct
  from the base spec's `soft 2D cutout` default used by the auto-rhyme
  pipeline, which is unchanged), a repeatable character sub-form
  (name / description / personality / voice dropdown), submits to
  `POST /shows`, renders the returned reference sheets via the asset-view
  proxy so the user can see what got frozen before using it.
- `TriggerRunForm.tsx` gains a show picker (`GET /shows`, "no show" default
  preserving today's one-off behavior).
- `apps/ui/app/p/[id]/page.tsx` gains a "Download video" button using the
  asset-view proxy (fetch → blob → save); the existing `<video>` preview
  (already broken against `local://` URIs, a pre-existing gap) is unchanged.

## 9. Test additions

`tests/unit/test_shows.py` (mark `amendment02`): `save_show`/`load_show`/
`list_shows` round-trip against real Postgres (`pg_required` fixture);
Creative Director's and `char_refs`'s skip-when-already-set behavior (stub
call-count assertions); a full-graph run with `show_id` set completing to
`REVIEW` with the loaded cast intact; unknown `show_id` and missing `db_url`
both raise.

`tests/unit/test_amendment02_dialogue.py`: `_scene_violations`'s speaker-match
check; `build_audio_concat_cmd`'s output shape (golden); a real-ffmpeg
integration test asserting exactly one narration `AssetRef` per scene for a
dialogue-bearing manifest.

`tests/unit/test_amendment02_api.py`: `POST /shows` auth/success/empty-cast
rejection, `GET /shows` listing, `POST /runs` with unknown/known `show_id`,
`GET /assets/view` auth/streaming/404.

## 10. Migration note for the coding agent

This amendment is additive to already-implemented WP1-8 + Amendment 01 code.
Concretely:
- `models.py`: extend `CharacterRef`, add `DialogueLine`, extend `Scene` and
  `VideoProject` — all additive fields with defaults.
- `db.py` + two new Alembic migrations: `ShowRow`, `save_show`/`load_show`/
  `list_shows`, `show_id` column on `projects`.
- New `util/character_prompts.py`, used by both `graph/nodes.py::char_refs`
  and the new `POST /shows` handler.
- `creative_director.py` and `graph/nodes.py::char_refs`: one skip-guard each.
- `pipeline.py`: `show_id` param threaded into the fresh-run branch.
- `director.py` + `prompts/director.md`: dialogue speaker validation + prompt
  section.
- `music_director.py` + `util/ffmpeg.py`: dialogue synthesis, fully contained.
- `apps/api/src/api/main.py`: `POST /shows`, `GET /shows`, `GET /assets/view`,
  `show_id` on `RunRequest`/`ProjectSummary`/`ProjectDetail`.
- `apps/worker/src/worker/main.py`: `--show-id` option.
- `apps/ui`: new `/shows/new` page, show picker, download button.
- No changes required to Videographer (WP3), Editor (WP5), Critic's routing
  logic (WP6), or Observability (WP8) beyond passing `show_id` through where
  the base spec already threads `project_id`.
