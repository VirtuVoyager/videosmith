from __future__ import annotations

from storysmith.models import CharacterRef

# Character reference sheets are model-sheet compositions (multiple views +
# an expression grid laid out side by side, see build_char_ref_prompt below)
# -- a portrait-oriented 9:16 canvas (the show's own StyleContract.aspect_ratio)
# crushes that layout. This is a fixed override used only for the reference
# image itself; it's never composited into the final video, only shown in the
# UI gallery and handed to Critic's vision-QA as the "what should this
# character look like" comparison image, so it doesn't need to match the
# video's aspect ratio at all.
CHAR_REF_ASPECT_RATIO = "3:2"


def build_char_ref_prompt(character: CharacterRef, art_style: str) -> str:
    """The character-reference-sheet prompt (§2.3) -- shared by
    graph/nodes.py::char_refs (per-project, LLM-authored casts) and
    apps/api's POST /shows (user-authored casts, Amendment 02), so both
    generate portraits from the identical prompt shape instead of two copies
    that can silently drift apart.

    Requests a professional character-turnaround-sheet layout (multiple full
    body views + an expression grid) rather than a single portrait -- the
    same reference image is what Critic's vision-QA compares every scene
    keyframe against for the character_consistency criterion, so more views
    of the character in one image gives it (and the human reviewing the
    avatar gallery) more to check consistency against than one static pose
    can. SPEC-GAP: flux-schnell (the current image_model) can't reliably
    render legible text, so this deliberately doesn't ask for on-image
    labels/hex-code swatches the way a hand-drawn model sheet would carry --
    only the visual structure (views + expressions), which diffusion models
    render far more reliably.
    """
    return (
        f"Character reference turnaround sheet: {character.description}, {art_style}. "
        "Professional animation model sheet layout on a plain white background: "
        "four full-body poses of the exact same character in a neutral standing "
        "pose, in a row -- front view, three-quarter view, side profile, and back "
        "view -- with perfectly consistent proportions, colors, and design across "
        "every pose. Below or beside the poses, a small grid of head-and-shoulders "
        "expression studies of the same character (neutral, talking, thoughtful, "
        "and one other clear expression). Even, shadowless studio lighting, sharp "
        "linework, no background clutter, no text."
    )
