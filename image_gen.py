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


_TRANSLATE_SYSTEM = (
    "Translate the following text to English. "
    "Output only the translation, no explanation, no preamble, no quotes."
)


def translate_to_english(client: NvidiaClient, text: str) -> str:
    """Translate *text* to English via LLM. Returns '' for empty input."""
    if not (text and text.strip()):
        return ""
    return client.chat(
        [
            {"role": "system", "content": _TRANSLATE_SYSTEM},
            {"role": "user", "content": text},
        ],
        model=DEFAULT_LLM,
        temperature=0.1,
        max_tokens=1500,
    ).strip()


def ensure_english_fields(client: NvidiaClient, project: Project, scene: Scene) -> None:
    """Populate empty *_en fields for project + scene by translating.

    Lazy-migration helper: old projects without *_en fields get their
    translations computed on first use. The project is saved by the caller
    after image generation, so translations are cached automatically.
    """
    if project.style_anchor and not project.style_anchor_en:
        project.style_anchor_en = translate_to_english(client, project.style_anchor)
    for cname in scene.characters:
        c = project.characters.get(cname)
        if c and c.description and not c.description_en:
            c.description_en = translate_to_english(client, c.description)
    if scene.prompt and not scene.prompt_en:
        scene.prompt_en = translate_to_english(client, scene.prompt)


def build_prompt(project: Project, scene: Scene) -> str:
    """Assemble the FLUX prompt from pre-stored English texts — no LLM call.

    Sections (in order):
      SCENE      — what to draw, most important, drives composition
      CHARACTERS — visual descriptions of the characters present
      STYLE      — rendering style, palette, medium, line work
    Falls back to the original (possibly non-English) text when *_en is empty.
    """
    sections: list[str] = []

    scene_text = (scene.prompt_en or scene.prompt).strip()
    if scene_text:
        sections.append(f"SCENE: {scene_text}")

    char_parts: list[str] = []
    for cname in scene.characters:
        c = project.characters.get(cname)
        if c:
            desc = (c.description_en or c.description).strip()
            if desc:
                char_parts.append(f"{c.name}: {desc}")
    if char_parts:
        sections.append("CHARACTERS: " + " | ".join(char_parts))

    style = (project.style_anchor_en or project.style_anchor).strip()
    if style:
        sections.append(f"STYLE: {style}")

    return " ### ".join(sections)


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
    ensure_english_fields(client, project, scene)
    full_prompt = build_prompt(project, scene)
    if extra_prompt_fragment.strip():
        full_prompt = f"{full_prompt}, {extra_prompt_fragment.strip()}"

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


def generate_style_anchor_image(
    client: NvidiaClient,
    project: Project,
    output_dir: Path,
    aspect_ratio: str = "16:9",
) -> tuple[Path, str]:
    """Generate the style-anchor image. Returns (image_path, full_prompt).

    The prompt is the style_anchor_en field (translated once and cached).
    If style_anchor_en is empty, it is translated from style_anchor now.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    if not project.style_anchor:
        raise RuntimeError("No style anchor text set — write one first.")
    if not project.style_anchor_en:
        project.style_anchor_en = translate_to_english(client, project.style_anchor)
    prompt = project.style_anchor_en or project.style_anchor
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
