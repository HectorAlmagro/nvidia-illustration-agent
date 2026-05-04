from __future__ import annotations
import json
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class Character:
    name: str
    description: str
    reference_images: list[str] = field(default_factory=list)


@dataclass
class Scene:
    id: int
    title: str
    prompt: str
    characters: list[str] = field(default_factory=list)
    image_path: Optional[str] = None
    versions: list[str] = field(default_factory=list)
    seed: Optional[int] = None
    review_notes: list[str] = field(default_factory=list)
    last_full_prompt: str = ""


@dataclass
class Project:
    name: str
    style_anchor: str = ""
    style_image: Optional[str] = None
    style_image_prompt: str = ""
    synopsis: str = ""
    characters: dict[str, Character] = field(default_factory=dict)
    scenes: list[Scene] = field(default_factory=list)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "name": self.name,
            "style_anchor": self.style_anchor,
            "style_image": self.style_image,
            "style_image_prompt": self.style_image_prompt,
            "synopsis": self.synopsis,
            "characters": {k: asdict(v) for k, v in self.characters.items()},
            "scenes": [asdict(s) for s in self.scenes],
        }
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False))

    @classmethod
    def load(cls, path: Path) -> "Project":
        data = json.loads(path.read_text())
        chars = {k: Character(**v) for k, v in data.pop("characters", {}).items()}
        scenes = [Scene(**s) for s in data.pop("scenes", [])]
        p = cls(**data)
        p.characters = chars
        p.scenes = scenes
        return p

    def scene_by_id(self, sid: int) -> Optional[Scene]:
        return next((s for s in self.scenes if s.id == sid), None)

    def images_with_character(self, name: str) -> list[str]:
        out: list[str] = []
        for s in self.scenes:
            if s.image_path and name in s.characters:
                out.append(s.image_path)
        return out
