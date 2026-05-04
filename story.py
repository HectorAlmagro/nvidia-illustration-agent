from __future__ import annotations
import json
import re
from typing import Optional
from nvidia_client import NvidiaClient
from state import Project, Character, Scene


SYSTEM_PROMPT = """You are a story-and-illustration co-writer. You help the user
develop a short illustrated story. You are concise, propose concrete options,
and adapt to feedback.

When the user wants to commit story structure, return STRICTLY valid JSON inside
a fenced ```json code block with this schema:

{
  "synopsis": "<one paragraph>",
  "style_anchor": "<visual style description: medium, palette, lighting, line work, mood>",
  "characters": [
    {"name": "<unique name>", "description": "<detailed visual description for image gen: age, build, clothing, distinctive features, color palette>"}
  ],
  "scenes": [
    {"id": 1, "title": "<short>", "characters": ["<name>"], "prompt": "<image prompt focused on visual content of THIS scene only — do not include style or character full descriptions, those are added automatically>"}
  ]
}

Rules:
- character `description` must be detailed enough that two artists would draw the same character.
- scene `prompt` is just the action/setting; style + character details are appended at render time.
- only emit the JSON block when the user explicitly confirms ("ok", "go", "lock it", "commit").
- otherwise, talk in plain text, brainstorm, ask focused questions.
"""


JSON_BLOCK_RE = re.compile(r"```json\s*(.*?)\s*```", re.DOTALL)


def extract_story_json(text: str) -> Optional[dict]:
    m = JSON_BLOCK_RE.search(text)
    if not m:
        return None
    try:
        return json.loads(m.group(1))
    except json.JSONDecodeError:
        return None


def apply_story_to_project(project: Project, data: dict) -> None:
    project.synopsis = data.get("synopsis", project.synopsis)
    project.style_anchor = data.get("style_anchor", project.style_anchor)

    for c in data.get("characters", []):
        name = c["name"]
        if name in project.characters:
            project.characters[name].description = c["description"]
        else:
            project.characters[name] = Character(
                name=name, description=c["description"]
            )

    existing_by_id = {s.id: s for s in project.scenes}
    new_scenes: list[Scene] = []
    for s in data.get("scenes", []):
        sid = s["id"]
        if sid in existing_by_id:
            old = existing_by_id[sid]
            old.title = s.get("title", old.title)
            old.prompt = s.get("prompt", old.prompt)
            old.characters = s.get("characters", old.characters)
            new_scenes.append(old)
        else:
            new_scenes.append(
                Scene(
                    id=sid,
                    title=s.get("title", f"Scene {sid}"),
                    prompt=s["prompt"],
                    characters=s.get("characters", []),
                )
            )
    project.scenes = sorted(new_scenes, key=lambda s: s.id)


class StoryConversation:
    def __init__(self, client: NvidiaClient, project: Project):
        self.client = client
        self.project = project
        self.history: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
        self._seed_history()

    def _seed_history(self) -> None:
        """Always seed with the project identity + current state, so every
        chat turn is grounded on THIS project (even when empty)."""
        ctx = {
            "project_name": self.project.name,
            "synopsis": self.project.synopsis or "(not set)",
            "style_anchor": self.project.style_anchor or "(not set)",
            "characters": [
                {"name": c.name, "description": c.description}
                for c in self.project.characters.values()
            ],
            "scenes": [
                {
                    "id": s.id,
                    "title": s.title,
                    "characters": s.characters,
                    "prompt": s.prompt,
                }
                for s in self.project.scenes
            ],
        }
        self.history.append(
            {
                "role": "system",
                "content": (
                    f"You are working on the user's project '{self.project.name}'. "
                    f"Always interpret the user's questions and ideas in the "
                    f"context of THIS project (do not reference other projects). "
                    f"Current project state:\n"
                    + json.dumps(ctx, indent=2, ensure_ascii=False)
                ),
            }
        )

    def turn(self, user_input: str) -> tuple[str, Optional[dict]]:
        self.history.append({"role": "user", "content": user_input})
        reply = self.client.chat(self.history, temperature=0.7)
        self.history.append({"role": "assistant", "content": reply})
        data = extract_story_json(reply)
        if data:
            apply_story_to_project(self.project, data)
        return reply, data
