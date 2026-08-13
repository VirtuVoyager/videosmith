# StorySmith — Amendment 01: Start-Frame Conditioned Scene Generation

Status: AMENDMENT. Layers on top of `HANDOFF_SPEC.md` (WP1-8), which is unchanged
and remains the base contract. This document describes only the diff. Where this
amendment is silent, the base spec governs. Apply after WP1-6 (already implemented);
touches WP2 (Director), adds a new node between Director and Videographer, and
touches WP3 (video adapters). Does not touch WP4/WP5/WP6/WP7/WP8 logic, only the
`Scene` shape they consume.

## 1. Problem observed (evidence, not theory)

Manual testing (Gemini video model, civic "zebra crossing" scene) showed pure
text-to-video generation cannot hold spatial layout across a multi-beat scene:
crosswalk orientation changed mid-shot, a second perpendicular crosswalk appeared,
the character teleported position as a state change (red→green light) occurred,
and the crossing led nowhere logically in the background. Character *appearance*
(color, outfit, style) stayed consistent — this is specifically a **spatial
grounding failure**, not a character-consistency failure. Root cause: t2v models
have no persistent scene representation; a prompt describing a sequence of events
in one shot asks the model to hallucinate continuity with nothing anchoring it.

## 2. Fix: image-to-video (i2v) as default, t2v as an explicit per-scene opt-out

Generate a fully composed still image per scene first (exact camera framing,
object placement, light/state) and condition video generation on that image as
the start frame. The video model then only has to animate motion within a fixed
frame, not invent geometry — a categorically easier and more reliable task.

**t2v is retained, not removed** — value: cheaper (skips the image-gen step),
appropriate for geometry-light shots (sky, water, abstract transitions, close-up
textures) where nothing spatial can go wrong, and serves as a resilience fallback
if the image provider fails. Default is i2v for any scene the Director judges
spatially/relationally complex; t2v is an explicit flag, not the fallback default.

## 3. Model changes (`storysmith-core/src/storysmith/models.py`)

Extend `Scene` (base spec §1.1) with these fields — additive, non-breaking:

```python
class SceneGenMode(StrEnum):
    T2V = "t2v"
    I2V = "i2v"

class Scene(BaseModel):
    # ... existing fields unchanged (index, duration_s, video_prompt, narration, transition) ...
    gen_mode: SceneGenMode = SceneGenMode.I2V
    scene_image_prompt: str | None = None
    # required when gen_mode == I2V; a fully composed still-image prompt:
    # exact camera framing, exact object/character placement, exact state
    # (light color, pose, etc). Must NOT describe motion or time passing.
    # video_prompt for an I2V scene must then describe ONLY motion within
    # that fixed frame (e.g. "the duckling looks left then right, camera
    # static, nothing else in frame moves") — no new objects, no camera
    # moves unless explicitly the point of the shot.
```

Add `AssetKind.SCENE_STILL = "scene_still"` to the existing `AssetKind` enum
(base spec §1.1), alongside `CHAR_IMAGE`.

Director validation (post-LLM code check, extends base spec §2.2 rules):
every scene with `gen_mode == I2V` must have non-empty `scene_image_prompt`;
reject/retry-correct otherwise. A `video_prompt` for an I2V scene containing
scene-setting nouns already present in `scene_image_prompt` (crude check: token
overlap above a threshold on nouns via a small stopword-filtered set) triggers
one corrective LLM round with the instruction "video_prompt must describe
motion only, not scene composition — that is already fixed by the start frame."

## 4. Director prompt changes (`prompts/director.md`)

Add explicit instructions:
- Default every scene to `gen_mode: i2v`.
- Set `gen_mode: t2v` only for shots with no named character-object spatial
  relationships to get wrong (sky, water texture, abstract transition, close-up
  on a single non-interactive element). If in doubt, use i2v.
- For every i2v scene, write two separate prompts:
  1. `scene_image_prompt` — a complete static composition. Must specify: camera
     angle/framing, exact position of every named object/character relative to
     frame (e.g. "left-of-center," "foreground," "background"), and current
     state of anything stateful (traffic light color, door open/closed). Must
     read like a single photograph description — no verbs implying change over
     time.
  2. `video_prompt` — motion only. What moves, what explicitly does NOT move
     ("camera remains static, background stays fixed" when true), and nothing
     about layout (already fixed by the image).
- Split any scene that previously implied a state change mid-shot (e.g. "waits
  then crosses when light turns green") into two consecutive scenes: one whose
  start frame captures the "before" state, one whose start frame captures the
  "after" state. Let the Editor's crossfade (base spec §5, existing behavior)
  bridge the cut. Do not ask one generation to depict the transition itself
  unless the specific video model in use supports end-frame conditioning
  (§6 below) — that capability is queried from adapter config, not assumed.

## 5. New graph node: `scene_stills` (extends base spec §1.4, §2.3)

Insert between `char_refs` and `videographer` in the graph:

`char_refs -> scene_stills -> videographer` (parallel branch to `music_director`
unchanged from base spec).

- For each scene with `gen_mode == I2V`: call `ImageGenPort.generate` with a
  prompt built from `scene_image_prompt` plus a reference-conditioning input —
  pass the relevant `CharacterRef.image_uri` bytes as a style/identity anchor
  if the underlying image adapter supports image-conditioned generation
  (Flux Kontext / IP-adapter-style editing endpoints do; plain Flux-schnell
  text-to-image does not — check adapter capability, see §6). If the image
  adapter used is text-only, fold character appearance description directly
  into `scene_image_prompt` text instead (Director already has the character
  description available and should always include it in the composition
  prompt regardless, as a redundancy — cheap insurance).
- Store result as `AssetKind.SCENE_STILL`, keyed
  `{project_id}/scene_{index}/still.png`. Set a new field on the in-memory
  scene-processing context (not on the persisted `Scene` model — keep that
  model prompt-only) mapping `scene_index -> still_asset_uri`, threaded to
  `videographer` via the existing `VideoProject.assets` list (query by kind +
  index, same pattern already used for scene videos).
- Idempotency: same content-hash-skip pattern as base spec §3.2, hash over
  `scene_image_prompt + character_ref_hash`.
- On Critic retry (base spec §6) for an i2v scene, regenerate the still first
  if the critique indicates a *composition* problem (wrong layout, wrong
  object placement); regenerate only the video if the critique indicates a
  *motion* problem (bad animation of an otherwise-correct frame). Critic must
  therefore classify failure type — add `failure_layer: "composition" | "motion" | "other"`
  to `QAReport` (base spec §1.1 model, additive field, default `"other"` for
  backward compatibility with existing WP6 code).

## 6. Video adapter changes (extends base spec §3.1a/§3.1b)

Both `video_fal.py` and `video_replicate.py`:
- `VideoGenPort.generate` signature already accepts `reference_image: bytes | None`
  (base spec §1.3) — for i2v scenes, this is now the **scene still**, not the
  character reference. No port signature change needed; the videographer node
  simply passes the scene-still bytes instead of (or in addition to, if the
  specific model accepts both) the character-ref bytes.
- Add a small adapter capability flag, not a port change: a module-level
  constant `SUPPORTS_END_FRAME: bool` per adapter/model combination, checked
  by the `director.py` validation (§4 above) before allowing a single scene to
  depict a state transition in one generation. Default `False` unless verified
  against the specific model id in settings.
- For t2v scenes (`gen_mode == T2V`): call exactly as base spec described,
  `reference_image=None`, `prompt=scene.video_prompt` only (no still generated,
  `scene_stills` node skips these scenes entirely — check `gen_mode` before
  calling `ImageGenPort`).

## 7. Settings additions (`.env.example`)

```bash
# --- Scene composition ---
SS_DEFAULT_SCENE_GEN_MODE=i2v        # i2v | t2v — Director's default when unsure
SS_SCENE_IMAGE_MODEL=black-forest-labs/flux-kontext-pro  # supports image-conditioning
```

## 8. Test additions

`tests/unit/test_amendment01_scene_stills.py` (mark `amendment01`):
1. `test_i2v_scene_requires_image_prompt` — Director validation rejects/corrects
   a scene with `gen_mode=i2v` and empty `scene_image_prompt`.
2. `test_t2v_scene_skips_still_generation` — `scene_stills` node makes zero
   `ImageGenPort` calls for t2v-flagged scenes (stub call-count assertion).
3. `test_videographer_uses_scene_still_not_char_ref_for_i2v` — stub port
   receives the scene-still bytes, not the character-ref bytes, for an i2v scene.
4. `test_critic_failure_layer_routes_regeneration` — `failure_layer=composition`
   triggers still regeneration before video regeneration; `failure_layer=motion`
   skips straight to video regeneration (stub call sequence assertion).
5. `test_video_prompt_motion_only_flags_layout_overlap` — a video_prompt
   containing scene-setting nouns already in scene_image_prompt triggers the
   corrective LLM round (stub LLM call-count assertion).

`tests/llm/test_scene_composition_quality.py` (deepeval, manual/nightly per
base spec §10.2): GEval criterion `MotionPromptIsMotionOnly` — penalize any
`video_prompt` describing objects, layout, or "new" scene elements rather than
pure motion/camera behavior of an already-fixed frame.

## 9. Migration note for the coding agent

This amendment is additive to already-implemented WP1-6 code. Concretely:
- `models.py`: add `SceneGenMode`, extend `Scene`, extend `AssetKind`, add
  `failure_layer` to `QAReport` — all additive fields with defaults, existing
  serialized state/checkpoints remain loadable.
- `director.py`: extend prompt + validation per §4.
- New file `agents/scene_stills.py` + corresponding graph node registration
  in `graph/build.py` (insert node + rewire two edges, per §5).
- `videographer.py`: change which asset it reads as `reference_image` per §6
  (small, localized diff — look up `SCENE_STILL` asset for the scene index
  when `gen_mode==i2v`, else `None`).
- `critic.py`: add `failure_layer` classification to its structured output
  schema and rubric prompt; extend the routing conditional edge per §5's last
  bullet.
- No changes required to Editor (WP5), Music Director (WP4), Publisher (WP7),
  or Observability (WP8) — they are unaffected by this amendment.
