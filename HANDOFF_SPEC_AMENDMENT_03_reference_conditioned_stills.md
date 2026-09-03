# StorySmith — Amendment 03: Reference-Conditioned Scene Stills

Status: AMENDMENT. Layers on top of `HANDOFF_SPEC.md` (WP1-8), Amendment 01
(stills-first scene composition), and Amendment 02 (user-authored shows),
all unchanged and remaining the base contract. This document describes only
the diff. Touches WP1 (`ports.py`, `ImageGenPort`), WP2/Amendment 01
(`scene_stills.py`), and the Replicate image adapter. Does not touch
Director, Critic, Videographer, Editor, or the graph topology.

## 1. Problem observed (evidence, not theory)

A real live episode ("Crocky & Roachy", a frozen two-cockroach show) showed
each scene rendering a visibly different-looking pair of characters --
a human with glasses and a fox with a bowtie in scene 0, two green
lizard-like creatures in scene 1, two grey lizard-like creatures in scene
2 -- despite Amendment 02's character-restatement validation (director.py)
confirming every scene's `scene_image_prompt` fully described both
characters' appearance in text every time.

Root cause, confirmed via research (not assumed): `flux-schnell` (the
still-image model, text-to-image only) has no memory between independent
generations. Each call samples fresh from the diffusion prior guided only
by the text embedding; natural language can't pin exact pixel-level
appearance the way an image can, so small variances in interpretation
compound into visibly different character designs on every call, even with
identical, fully-restated descriptions. This is a well-documented problem
in the field (StoryMaker, CharaConsist, StorySync and other 2025-26 papers
on consistent character generation).

## 2. Fix: condition scene generation on the frozen reference sheet

Industry-standard fix, confirmed live before committing to it: condition
generation on an actual reference image via an image-editing model, not
text alone. `black-forest-labs/flux-kontext-pro` (`input_image` + a text
edit instruction) was already anticipated in the base settings
(`scene_image_model`) but never wired in. Verified live: feeding a
character's real frozen reference sheet as `input_image` produced a new
scene with identical face/glasses/vest/colors to the reference -- night
and day versus the text-only result.

Multi-character scenes (this project's actual case, ≤2 characters per
style contract) need a second trick, since Kontext -- like most
single-image-conditioned editors -- takes exactly one `input_image`, not
several: stitch every character's reference sheet into one composite
("reference grid") image and feed that as the single input. Also a
documented workaround (e.g. Flux 2 Klein's dedicated grid-stitching node),
and verified live here too -- a stitched Crocky+Roachy grid produced both
characters correctly composed together in a brand-new cafe scene, in one
API call.

## 3. Port change (`ports.py`)

```python
class ImageGenPort(Protocol):
    async def generate(
        self, *, prompt: str, aspect_ratio: str, reference_image: bytes | None = None
    ) -> tuple[bytes, float]: ...
```

Additive default (`None`) -- every existing caller (`char_refs`, `POST
/shows`'s avatar generation) is unaffected and keeps using pure
text-to-image, since generating the *first* reference sheet has no prior
appearance to condition on. Every port implementation (`StubImageGen`,
`RecordedImageGen`, `ReplicateImageGen`) updated to accept the parameter.

## 4. Adapter (`image_replicate.py`)

Same model-selection-by-argument-presence pattern `video_replicate.py`
already uses for i2v/t2v: no `reference_image` routes to `image_model`
(flux-schnell); a `reference_image` present routes to `scene_image_model`
(flux-kontext-pro), sent as a base64 data URI in Kontext's `input_image`
field, with `aspect_ratio` passed explicitly (not Kontext's
`match_input_image` default) -- the reference sheet is a wide 3:2
turnaround, but the scene still itself must be the show's actual video
aspect ratio (9:16).

## 5. `scene_stills.py` changes

- `_build_reference_image(style, ports)`: fetches every character's frozen
  `image_uri` via `StoragePort`, returns `None` if none exist yet (falls
  back to Amendment 01/02's text-only behavior unchanged), the single image
  unmodified if exactly one character has a reference, or a stitched
  composite (`_stitch_horizontally`) if 2+. Built once per `run()` call, not
  per scene -- the same frozen cast conditions every scene in a pass.
- `_stitch_horizontally(images)`: Pillow-based composite, each image
  resized to the shortest one's height first so the grid isn't lopsided.
- `_generate_one` passes the reference image through to
  `ImageGenPort.generate` and folds it into the content-hash idempotency
  check (`sha256_hex(model_id, prompt, ref_hash)`) alongside the model id
  actually used -- a scene generated with vs. without a reference image (or
  against a different reference) must never be treated as the same cached
  request.
- Character descriptions stay in the prompt text too (Director already
  writes them there per Amendment 02's restatement check) -- redundant with
  the reference image, but cheap insurance, and still the only signal for
  the rare case a scene generates before `char_refs`/`POST /shows` has
  produced a reference yet.

## 6. Cost

`flux-kontext-pro` is priced higher than `flux-schnell` ($0.04/image
estimate vs. $0.003) -- a real, small per-episode cost increase (roughly
+$0.18 across 5 scenes) in exchange for actually solving the consistency
problem instead of silently shipping visibly wrong characters.

## 7. Test additions

`tests/unit/test_amendment03_reference_conditioning.py` (mark
`amendment03`): `_stitch_horizontally`'s output dimensions for
differently-sized inputs; `_build_reference_image`'s three cases (no
avatars, one avatar, two avatars stitched); `_generate_one` passing the
reference image through to the port; content-hash divergence with vs.
without a reference image. `test_amendment01_scene_stills.py`'s existing
image-gen test doubles updated to accept the new keyword-only parameter.

## 8. Migration note for the coding agent

Additive throughout -- no existing checkpoints, tests, or call sites break
except test doubles with an explicit (non-`**kwargs`) `generate()`
signature, which needed the new parameter added (see `git log` for the
exact list touched). No settings changes needed: `scene_image_model` already
existed in the base config surface, just unused until now.
