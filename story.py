from __future__ import annotations
import json
import re
import sys
from dataclasses import dataclass, field
from typing import Optional
from nvidia_client import NvidiaClient
from state import Project, Character, Scene


@dataclass
class ChangedFields:
    synopsis: bool = False
    style_anchor: bool = False
    characters: list[str] = field(default_factory=list)
    scene_ids: list[int] = field(default_factory=list)

    def any(self) -> bool:
        return bool(self.synopsis or self.style_anchor or self.characters or self.scene_ids)


SYSTEM_PROMPT = """You are a story-and-illustration co-writer. You help the user
develop a short illustrated story. You are concise, propose concrete options,
and adapt to feedback.

When the user wants to commit changes, return STRICTLY valid JSON inside a
fenced ```json code block containing ONLY the fields that actually changed.
Do NOT include fields that were not modified.

Allowed top-level keys: "synopsis", "style_anchor", "characters", "scenes".

- To update synopsis or style_anchor, include just that key with the new value.
- To update characters, include only the characters that changed (by name).
  Existing characters not listed remain untouched.
- To update scenes, include only the scenes that changed (by id).
  Existing scenes not listed remain untouched.

Schema for a character entry:
  {"name": "<unique name>", "description": "<detailed visual description: age, build, clothing, distinctive features, color palette>"}

Schema for a scene entry:
  {"id": <int>, "title": "<short>", "characters": ["<name>"], "prompt": "<visual content of THIS scene only — do not include style or character full descriptions>"}

Rules:
- character `description` must be detailed enough that two artists would draw the same character.
- scene `prompt` is just the action/setting; style + character details are appended at render time.
- only emit the JSON block when the user explicitly confirms ("ok", "go", "lock it", "commit", "save", "apply").
- otherwise, talk in plain text, brainstorm, ask focused questions.
- NEVER emit the full project JSON just because it was seeded in context — only emit what the user asked to change.

CRITICAL — LANGUAGE RULE:
All text values inside the JSON (synopsis, style_anchor, character descriptions,
scene prompts) MUST be written in ENGLISH, regardless of the language of the
conversation. Translate to English before writing each value into the JSON.
The conversation itself can be in any language, but every JSON field value
must be in English so it can be used directly as an image-generation prompt.
"""


JSON_BLOCK_RE = re.compile(r"```json\s*(.*?)\s*```", re.DOTALL)


def extract_story_json(text: str) -> Optional[dict]:
    m = JSON_BLOCK_RE.search(text)
    if not m:
        return None
    try:
        data = json.loads(m.group(1))
        if isinstance(data, str):
            try:
                data = json.loads(data)
            except json.JSONDecodeError:
                return None
        return data if isinstance(data, dict) else None
    except json.JSONDecodeError:
        return None


def _log_story_warning(message: str) -> None:
    print(f"[story] WARNING {message}", file=sys.stderr)


def _iter_character_updates(raw) -> list[dict]:
    if not raw:
        return []
    if isinstance(raw, dict):
        items = []
        for name, value in raw.items():
            if isinstance(value, dict):
                item = dict(value)
                item.setdefault("name", name)
                items.append(item)
            elif isinstance(value, str):
                items.append({"name": name, "description": value})
            else:
                _log_story_warning(f"Skipping character '{name}' with unsupported value type {type(value).__name__}")
        return items
    if isinstance(raw, list):
        items = []
        for item in raw:
            if isinstance(item, dict):
                items.append(item)
            elif isinstance(item, str):
                _log_story_warning(f"Skipping character string entry: {item[:120]!r}")
            else:
                _log_story_warning(f"Skipping character entry with unsupported type {type(item).__name__}")
        return items
    _log_story_warning(f"Skipping characters payload with unsupported type {type(raw).__name__}")
    return []


def _iter_scene_updates(raw) -> list[dict]:
    if not raw:
        return []
    if isinstance(raw, dict):
        items = []
        for sid, value in raw.items():
            if isinstance(value, dict):
                item = dict(value)
                item.setdefault("id", sid)
                items.append(item)
            else:
                _log_story_warning(f"Skipping scene '{sid}' with unsupported value type {type(value).__name__}")
        return items
    if isinstance(raw, list):
        items = []
        for item in raw:
            if isinstance(item, dict):
                items.append(item)
            elif isinstance(item, str):
                _log_story_warning(f"Skipping scene string entry: {item[:120]!r}")
            else:
                _log_story_warning(f"Skipping scene entry with unsupported type {type(item).__name__}")
        return items
    _log_story_warning(f"Skipping scenes payload with unsupported type {type(raw).__name__}")
    return []


def _coerce_scene_id(value) -> Optional[int]:
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return None


def apply_story_to_project(project: Project, data: dict) -> ChangedFields:
    """Merge a partial JSON update into the project. Only fields present in
    *data* are touched; everything else is left as-is.
    Returns a ChangedFields record of what actually changed."""
    changed = ChangedFields()

    if "synopsis" in data:
        if isinstance(data["synopsis"], str):
            project.synopsis = data["synopsis"]
            # _en will be filled by caller via batch_translate
            changed.synopsis = True
        else:
            _log_story_warning(f"Skipping synopsis with unsupported type {type(data['synopsis']).__name__}")

    if "style_anchor" in data:
        if isinstance(data["style_anchor"], str):
            project.style_anchor = data["style_anchor"]
            changed.style_anchor = True
        else:
            _log_story_warning(f"Skipping style_anchor with unsupported type {type(data['style_anchor']).__name__}")

    for c in _iter_character_updates(data.get("characters")):
        if not isinstance(c, dict):
            _log_story_warning(f"Skipping malformed character entry: {type(c).__name__}")
            continue
        name = c.get("name")
        desc = c.get("description")
        if not isinstance(name, str) or not isinstance(desc, str):
            _log_story_warning(f"Skipping character entry with bad fields: {c!r}")
            continue
        if name in project.characters:
            project.characters[name].description = desc
        else:
            project.characters[name] = Character(name=name, description=desc)
        changed.characters.append(name)

    existing_by_id = {s.id: s for s in project.scenes}
    for s in _iter_scene_updates(data.get("scenes")):
        if not isinstance(s, dict):
            _log_story_warning(f"Skipping malformed scene entry: {type(s).__name__}")
            continue
        sid = _coerce_scene_id(s.get("id"))
        if sid is None:
            _log_story_warning(f"Skipping scene entry with bad id: {s!r}")
            continue
        prompt_changed = False
        if sid in existing_by_id:
            old = existing_by_id[sid]
            if isinstance(s.get("title"), str):
                old.title = s["title"]
            if isinstance(s.get("prompt"), str):
                old.prompt = s["prompt"]
                prompt_changed = True
            if isinstance(s.get("characters"), list):
                old.characters = [x for x in s["characters"] if isinstance(x, str)]
        else:
            chars = s.get("characters", [])
            if not isinstance(chars, list):
                chars = []
            prompt_changed = isinstance(s.get("prompt"), str)
            project.scenes.append(
                Scene(
                    id=sid,
                    title=s["title"] if isinstance(s.get("title"), str) else f"Scene {sid}",
                    prompt=s["prompt"] if prompt_changed else "",
                    characters=[x for x in chars if isinstance(x, str)],
                )
            )
        if prompt_changed:
            changed.scene_ids.append(sid)
    project.scenes = sorted(project.scenes, key=lambda s: s.id)
    return changed


# ── triage prompt ─────────────────────────────────────────────────────────────

TRIAGE_PROMPT = """You are a routing assistant for a story co-writer.
Given a short project overview and a user message, return ONLY a JSON object
(no markdown fences, no explanation) indicating which project data you need
to answer the user:

{"characters": ["<name>", ...], "scene_ids": [<int>, ...], "fields": ["synopsis", "style_anchor"]}

- List only the characters whose full descriptions are needed.
- List only the scene ids whose full prompts are needed.
- List only top-level fields (synopsis, style_anchor) if needed.
- If the user message requires no project data at all, return {}.
- If you are uncertain, list everything."""


class StoryConversation:
    def __init__(self, client: NvidiaClient, project: Project):
        self.client = client
        self.project = project
        # history stores only the system prompt + user/assistant turns;
        # project data is injected ephemerally per turn, never stored.
        self.history: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]

    def _overview(self) -> str:
        """Lightweight summary: no full descriptions."""
        chars = [{"name": c.name} for c in self.project.characters.values()]
        scenes = [{"id": s.id, "title": s.title} for s in self.project.scenes]
        synopsis_preview = (self.project.synopsis or "")[:200]
        ctx = {
            "project": self.project.name,
            "synopsis_preview": synopsis_preview or "(not set)",
            "style_anchor_set": bool(self.project.style_anchor),
            "characters": chars,
            "scenes": scenes,
        }
        return json.dumps(ctx, ensure_ascii=False)

    def _triage(self, user_input: str) -> dict:
        """Fast call to determine which project data is needed.
        Returns a dict like {"characters": [...], "scene_ids": [...], "fields": [...]}.
        Falls back to full context on parse failure.
        """
        messages = [
            {"role": "system", "content": TRIAGE_PROMPT},
            {
                "role": "user",
                "content": f"Project overview:\n{self._overview()}\n\nUser message:\n{user_input}",
            },
        ]
        try:
            raw = self.client.chat(messages, temperature=0, max_tokens=150)
            # strip possible code fences
            raw = raw.strip()
            if raw.startswith("```"):
                raw = re.sub(r"```[a-z]*\n?", "", raw).replace("```", "").strip()
            needs = json.loads(raw)
            if not isinstance(needs, dict):
                raise ValueError("triage returned non-dict")
            # normalise scene_ids to int
            needs["scene_ids"] = [
                int(x) for x in needs.get("scene_ids", []) if str(x).isdigit() or isinstance(x, int)
            ]
            print(f"[story] triage → {needs}", file=sys.stderr)
            return needs
        except Exception as exc:
            print(f"[story] triage parse failed ({exc}), loading full context", file=sys.stderr)
            return {
                "characters": list(self.project.characters.keys()),
                "scene_ids": [s.id for s in self.project.scenes],
                "fields": ["synopsis", "style_anchor"],
            }

    def _build_context(self, needs: dict) -> str:
        """Build a system message string with only the requested project data."""
        parts: list[str] = [f"Project: {self.project.name}"]
        fields = needs.get("fields", [])
        if "synopsis" in fields and self.project.synopsis:
            parts.append(f"Synopsis: {self.project.synopsis}")
        if "style_anchor" in fields and self.project.style_anchor:
            parts.append(f"Style anchor: {self.project.style_anchor}")
        char_names = needs.get("characters", [])
        if char_names:
            char_lines = []
            for name in char_names:
                c = self.project.characters.get(name)
                if c:
                    char_lines.append(f"  {c.name}: {c.description}")
            if char_lines:
                parts.append("Characters:\n" + "\n".join(char_lines))
        scene_ids = needs.get("scene_ids", [])
        if scene_ids:
            scene_lines = []
            for sid in scene_ids:
                s = self.project.scene_by_id(int(sid))
                if s:
                    scene_lines.append(
                        f"  Scene {s.id} '{s.title}' (characters: {', '.join(s.characters) or 'none'}): {s.prompt}"
                    )
            if scene_lines:
                parts.append("Scenes:\n" + "\n".join(scene_lines))
        return "\n".join(parts)

    def turn(self, user_input: str) -> tuple[str, Optional[dict], ChangedFields]:
        """Process one user turn.
        1. Triage: fast call to decide which data to load.
        2. Action: call with history + ephemeral context + user message.
        Returns (reply, raw_data_or_None, ChangedFields).
        """
        needs = self._triage(user_input)
        ctx_text = self._build_context(needs)

        # Build the messages for the action call:
        # history (system prompt + prior turns) + ephemeral context + user message
        ephemeral_ctx = {
            "role": "system",
            "content": (
                f"Current project context (use this to answer the user):\n{ctx_text}"
            ),
        }
        action_messages = self.history + [ephemeral_ctx, {"role": "user", "content": user_input}]

        reply = self.client.chat(action_messages, temperature=0.7)

        # Store only bare user + assistant turns in history (no descriptions)
        self.history.append({"role": "user", "content": user_input})
        self.history.append({"role": "assistant", "content": reply})

        data = extract_story_json(reply)
        changed = ChangedFields()
        if data:
            changed = apply_story_to_project(self.project, data)
        return reply, data, changed
