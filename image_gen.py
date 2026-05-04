from __future__ import annotations
import io
import random
from pathlib import Path
from typing import Optional
from PIL import Image, ImageStat
from nvidia_client import (
    NvidiaClient,
    DEFAULT_IMAGE_MODEL,
    DEFAULT_KONTEXT_MODEL,
    DEFAULT_LLM,
)
from state import Project, Scene


def _looks_blank(img_bytes: bytes, dark_threshold: float = 5.0) -> bool:
    """True if image is almost entirely black/blank — usually a content-filter trip."""
    try:
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        stat = ImageStat.Stat(img)
        return all(c < dark_threshold for c in stat.mean)
    except Exception:
        return False


PROMPT_ENHANCER_SYSTEM = """You convert a structured scene brief into one rich,
natural-language prompt for FLUX.1 (a text-to-image model).

OUTPUT LANGUAGE: always English, regardless of the input language. FLUX is
English-trained and produces sharper images from English prompts.

STRUCTURE — write a single paragraph in this order:
1. SETTING FIRST. Open with a concrete description of the physical location
   where the scene takes place RIGHT NOW (interior or exterior, key props,
   time of day, light, atmosphere). Be specific so the model anchors the
   scene in that place.
2. CHARACTERS IN ACTION. Place every character listed under CHARACTERS
   PRESENT into that setting, weaving their full visual description (age,
   build, hair, eyes, distinctive features, exact clothing in THIS scene)
   inline as part of the narration. State what each one is doing, their
   posture, their expression.
3. Cinematography: shot type / framing, camera angle, lens feel, lighting
   direction and quality, color palette.
4. Rendering style at the very end: medium, line work, level of stylization,
   color treatment.

HARD RULES:
- Output ONE paragraph of flowing English prose. No tag lists, no JSON, no
  labels, no preamble, no quotes, no markdown — just the paragraph.
- Preserve EVERY concrete detail from the scene brief: action, props,
  expressions, mood, who is wearing what, what is or is not in frame.
- Honor negatives — convert them into explicit positives ("X is dressed in
  everyday clothes, not swimwear", "Z stays out of the shot").
- Do NOT mention destinations, future actions, or places the characters are
  ABOUT TO go. Only describe what is visible in the current frame. Example:
  if the brief says "she gets ready to go to the pool", the prompt must show
  the bedroom / bathroom where she is getting ready, NOT the pool.
- Treat the STYLE GUIDE as RENDERING STYLE ONLY (medium, palette, line work,
  mood). If the style guide mentions subject matter (e.g. "water", "pool",
  "underwater scenes"), IGNORE that subject matter unless the scene brief
  itself places the action there. Use only the style/palette/medium cues.
- Use concrete visual nouns. Avoid abstract narrative words ("emotion",
  "story", "atmosphere of growth").
- Length: 500-900 characters of dense prose.
- Output ONLY the prompt paragraph.

CHARACTER COUNT & DIFFERENTIATION (critical for FLUX):
- State the EXACT number of people/characters visible at the very start of
  the character section. E.g. "Two figures occupy the scene: …"
- When characters have different ages (child vs adult), STRONGLY emphasize
  body-size contrast: use explicit terms like "tiny toddler barely reaching
  waist height" vs "tall adult woman", "small child" vs "full-grown woman".
  Mention relative heights. FLUX often renders all figures the same size
  unless you are extremely explicit about proportions.
- Never describe a character by age number alone — always pair it with a
  concrete size/proportion cue (e.g. "a very small 3-year-old toddler").

MIRRORS & REFLECTIONS (critical — FLUX cannot handle these correctly):
- NEVER write "standing in front of a mirror" or "looking at her reflection"
  or similar. FLUX will render duplicate/triplicate figures.
- Instead, reframe the composition: describe the character facing the camera
  directly (frontal view), as if the viewer IS the mirror. The mirror itself
  must NOT be mentioned in the prompt. If the brief says "we see her as if
  we were her reflection", translate this to "facing the viewer directly,
  frontal medium shot" — do NOT reference a mirror at all."""


def enhance_prompt(client: NvidiaClient, project: Project, scene: Scene) -> str:
    """Use LLM to compose a rich natural-language FLUX prompt from project + scene."""
    char_lines = []
    for cname in scene.characters:
        c = project.characters.get(cname)
        if c:
            char_lines.append(f"- {c.name}: {c.description}")
    user = (
        f"STYLE GUIDE (rendering style only — palette / medium / line work / "
        f"mood; ignore any subject-matter words here unless the scene below "
        f"explicitly places the action there):\n"
        f"{project.style_anchor or '(unspecified)'}\n\n"
        f"CHARACTERS PRESENT IN SCENE (integrate every visual detail inline):\n"
        + ("\n".join(char_lines) if char_lines else "(none)")
        + f"\n\nSCENE BRIEF (this is the ONLY source of truth for where the "
        f"action takes place and what is visible — preserve every detail "
        f"including negatives, and do NOT depict any future destination "
        f"mentioned in the brief):\n"
        f"{scene.prompt}\n\n"
        f"Write the FLUX prompt now as one rich English paragraph, opening "
        f"with the present setting."
    )
    return client.chat(
        [
            {"role": "system", "content": PROMPT_ENHANCER_SYSTEM},
            {"role": "user", "content": user},
        ],
        model=DEFAULT_LLM,
        temperature=0.4,
        max_tokens=700,
    ).strip()


def pick_reference(project: Project, scene: Scene) -> Optional[str]:
    for char_name in scene.characters:
        imgs = project.images_with_character(char_name)
        imgs = [i for i in imgs if i != scene.image_path]
        if imgs:
            return imgs[-1]
    if project.style_image and project.style_image != scene.image_path:
        return project.style_image
    return None


def generate_scene_image(
    client: NvidiaClient,
    project: Project,
    scene: Scene,
    output_dir: Path,
    use_kontext: bool = False,
    seed: Optional[int] = None,
    aspect_ratio: str = "16:9",
    cfg_scale: float = 4.5,
    steps: int = 50,
    extra_prompt_fragment: str = "",
) -> tuple[Path, str]:
    """Generate or regenerate a scene image. Returns (image_path, full_prompt)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    enhanced = enhance_prompt(client, project, scene)
    full_prompt = enhanced
    if extra_prompt_fragment.strip():
        full_prompt = f"{enhanced}, {extra_prompt_fragment.strip()}"

    ref = pick_reference(project, scene) if use_kontext else None
    chosen_seed = (
        seed if seed is not None else (scene.seed or random.randint(1, 2**31 - 1))
    )

    img_bytes: bytes
    if ref:
        try:
            img_bytes = client.generate_image(
                prompt=full_prompt,
                model=DEFAULT_KONTEXT_MODEL,
                seed=chosen_seed,
                ref_image_path=ref,
            )
        except RuntimeError as e:
            print(f"[kontext fallback] {e}")
            img_bytes = client.generate_image(
                prompt=full_prompt,
                model=DEFAULT_IMAGE_MODEL,
                seed=chosen_seed,
                aspect_ratio=aspect_ratio,
                cfg_scale=cfg_scale,
                steps=steps,
            )
    else:
        img_bytes = client.generate_image(
            prompt=full_prompt,
            model=DEFAULT_IMAGE_MODEL,
            seed=chosen_seed,
            aspect_ratio=aspect_ratio,
            cfg_scale=cfg_scale,
            steps=steps,
        )

    if _looks_blank(img_bytes):
        raise RuntimeError(
            "Image came back blank/black — the NVIDIA model likely refused the "
            "prompt (safety filter). Rephrase the scene or character "
            "descriptions to avoid sensitive content and retry."
        )

    version_n = len(scene.versions) + 1
    out_path = output_dir / f"scene_{scene.id:02d}_v{version_n:02d}.png"
    out_path.write_bytes(img_bytes)

    if scene.image_path:
        scene.versions.append(scene.image_path)
    scene.image_path = str(out_path)
    scene.seed = chosen_seed
    scene.last_full_prompt = full_prompt
    return out_path, full_prompt


STYLE_PROMPT_SYSTEM = """You are a faithful translator from any language into
English for FLUX.1 image-prompt use.

YOUR JOB: take the user's style description and produce the equivalent
description in English. This is a TRANSLATION + REPHRASING task, not a
summarization task.

HARD RULES:
- Output ONE English paragraph (or two short paragraphs if the input has
  clearly distinct ideas). No labels, no JSON, no preamble, no quotes,
  no markdown — just the prose.
- DO NOT SHORTEN, SUMMARIZE OR COMPRESS the input. Every single concrete
  detail in the input — every adjective, every example, every property —
  must appear in the output. If the input lists examples (e.g. "like beach
  sand, coral reef rock, water surface"), the output must keep those exact
  examples. If the input describes eyes, lighting, line quality, palette,
  textures, mood, etc., each of those descriptions must survive in full.
- The output should be ROUGHLY THE SAME LENGTH as the input (within 20%).
  If the input is 700 characters, the output must be ~700 characters.
- Do NOT add anything that is not in the input. No new subjects, no new
  characters, no new places, no new palette terms, no new mood words.
- Use concrete visual nouns; the wording should read naturally for a FLUX
  prompt, but DETAIL DENSITY MUST BE PRESERVED.
- Output ONLY the translated prose."""


def style_anchor_prompt(client: NvidiaClient, project: Project) -> str:
    """Translate project.style_anchor into a faithful English FLUX prompt.

    Length scales with the input — we don't want a summary, we want every
    detail preserved in English.
    """
    style_text = (project.style_anchor or "").strip()
    if not style_text:
        raise RuntimeError("No style anchor text set — write one first.")
    # Allow ~3 tokens per source word + headroom, capped reasonably.
    src_words = len(style_text.split())
    max_tok = max(400, min(1500, src_words * 4 + 200))
    return client.chat(
        [
            {"role": "system", "content": STYLE_PROMPT_SYSTEM},
            {"role": "user", "content": style_text},
        ],
        model=DEFAULT_LLM,
        temperature=0.2,
        max_tokens=max_tok,
    ).strip()


def generate_style_anchor_image(
    client: NvidiaClient,
    project: Project,
    output_dir: Path,
    aspect_ratio: str = "16:9",
) -> tuple[Path, str]:
    """Generate the style-anchor image. Returns (image_path, full_prompt).

    The prompt is built only from project.style_anchor (translated to English
    by the LLM); nothing else is appended.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    prompt = style_anchor_prompt(client, project)
    seed = random.randint(1, 2**31 - 1)
    img_bytes = client.generate_image(
        prompt=prompt,
        model=DEFAULT_IMAGE_MODEL,
        seed=seed,
        aspect_ratio=aspect_ratio,
        cfg_scale=4.5,
    )
    if _looks_blank(img_bytes):
        raise RuntimeError(
            "Style anchor came back blank/black — likely a safety-filter trip. "
            "Tone down the style description and retry."
        )
    out_path = output_dir / "style_anchor.png"
    out_path.write_bytes(img_bytes)
    project.style_image = str(out_path)
    project.style_image_prompt = prompt
    return out_path, prompt
