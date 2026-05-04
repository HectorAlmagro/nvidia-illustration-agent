from __future__ import annotations
import shutil
from pathlib import Path
from dotenv import load_dotenv
import gradio as gr

from nvidia_client import NvidiaClient
from state import Project
from story import StoryConversation
from image_gen import generate_scene_image, generate_style_anchor_image, translate_to_english


load_dotenv()
BASE = Path(__file__).parent
PROJECTS_DIR = BASE / "projects"
PROJECTS_DIR.mkdir(exist_ok=True)
OUTPUT_BASE = BASE / "output"
OUTPUT_BASE.mkdir(exist_ok=True)

client = NvidiaClient()

PENCIL = "✎"
EMPTY_MD = "_(empty — click ✎ to edit, or use the chat)_"
NO_PROJECT_MD = "_No project loaded._"


# ---------- helpers ----------

def _project_path(name: str) -> Path:
    return PROJECTS_DIR / f"{name}.json"


def _output_dir(name: str) -> Path:
    p = OUTPUT_BASE / name
    p.mkdir(parents=True, exist_ok=True)
    return p


def _save(project: Project | None) -> None:
    if project is None:
        return
    project.save(_project_path(project.name))


def _rebase_paths(project: Project) -> bool:
    out_dir = _output_dir(project.name)
    out_resolved = out_dir.resolve()
    changed = False

    def _fix(p: str | None) -> str | None:
        nonlocal changed
        if not p:
            return p
        path = Path(p)
        try:
            inside = out_resolved in path.resolve().parents or path.resolve() == out_resolved
        except OSError:
            inside = False
        if path.exists() and inside:
            return p
        candidate = out_dir / path.name
        if candidate.exists():
            changed = True
            return str(candidate)
        changed = True
        return None

    project.style_image = _fix(project.style_image)
    for s in project.scenes:
        s.image_path = _fix(s.image_path)
        s.versions = [v for v in (_fix(v) for v in s.versions) if v]
    return changed


def _list_projects() -> list[str]:
    return sorted([p.stem for p in PROJECTS_DIR.glob("*.json")])


def _scenes_rows(project: Project | None, selected_id: int | None = None) -> list[list]:
    if not project:
        return []
    return [
        [
            "▶" if (selected_id is not None and s.id == selected_id) else "",
            s.id,
            s.title,
            ", ".join(s.characters),
            "yes" if (s.image_path and Path(s.image_path).exists()) else "no",
        ]
        for s in project.scenes
    ]


def _ref_images_for(project: Project | None, char_name: str | None) -> list[str]:
    if not project or not char_name:
        return []
    return [p for p in project.images_with_character(char_name) if Path(p).exists()]


def _safe_path(p: str | None) -> str | None:
    if not p:
        return None
    return p if Path(p).exists() else None


def _scene_versions(project: Project | None, scene_id) -> list[str]:
    if not project or scene_id is None:
        return []
    scene = project.scene_by_id(int(scene_id))
    if not scene:
        return []
    paths: list[str] = [p for p in scene.versions if p != scene.image_path]
    if scene.image_path:
        paths.append(scene.image_path)
    seen: set[str] = set()
    out: list[str] = []
    for p in paths:
        if p in seen or not Path(p).exists():
            continue
        seen.add(p)
        out.append(p)
    return out


def _md_or_empty(text: str | None) -> str:
    return text if (text and text.strip()) else EMPTY_MD


def _has_image(project, scene_id) -> bool:
    if not project or scene_id is None:
        return False
    s = project.scene_by_id(int(scene_id))
    return bool(s and _safe_path(s.image_path))


def _redo_btn_update(project, scene_id):
    """Update for redo_btn — disabled if scene has no image."""
    return gr.update(interactive=_has_image(project, scene_id))


def _version_btn_updates(project, scene_id):
    """Updates for delete_version_btn and set_current_btn — disabled if no versions."""
    has_versions = bool(_scene_versions(project, scene_id))
    return gr.update(interactive=has_versions), gr.update(interactive=has_versions)


# ---------- inline edit visibility helpers ----------

def _show_edit():
    return gr.update(visible=False), gr.update(visible=True), gr.update(visible=True)


def _hide_edit():
    return gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)


# ---------- callbacks: project lifecycle ----------

def cb_load(name: str):
    name = (name or "").strip()
    if not name:
        return _empty_load_outputs()
    path = _project_path(name)
    if not path.exists():
        gr.Warning(f"No project '{name}'")
        return _empty_load_outputs()
    project = Project.load(path)
    if _rebase_paths(project):
        _save(project)
    convo = StoryConversation(client, project)
    return _hydrate_outputs(project, convo)


def cb_create(new_name: str):
    new_name = (new_name or "").strip()
    if not new_name:
        gr.Warning("Type a project name first")
        return _empty_load_outputs() + (gr.update(), gr.update())
    if _project_path(new_name).exists():
        gr.Warning(f"Project '{new_name}' already exists")
        return _empty_load_outputs() + (gr.update(), gr.update())
    project = Project(name=new_name)
    _save(project)
    convo = StoryConversation(client, project)
    gr.Info(f"Created '{new_name}'")
    return _hydrate_outputs(project, convo) + (
        gr.update(choices=_list_projects(), value=new_name),
        gr.update(value=""),
    )


def cb_delete_confirm(project, confirm_text: str):
    if not project:
        gr.Warning("No project loaded")
        return _empty_load_outputs() + (gr.update(choices=_list_projects()), gr.update(value=""))
    if confirm_text.strip() != project.name:
        gr.Warning(f"Type '{project.name}' exactly to confirm")
        return _empty_load_outputs() + (gr.update(), gr.update(value=confirm_text))
    name = project.name
    path = _project_path(name)
    if path.exists():
        path.unlink()
    out_dir = OUTPUT_BASE / name
    if out_dir.exists():
        shutil.rmtree(out_dir, ignore_errors=True)
    gr.Info(f"Deleted '{name}'")
    return _empty_load_outputs() + (gr.update(choices=_list_projects(), value=None), gr.update(value=""))


def _empty_load_outputs():
    return (
        None,                                       # project_state
        None,                                       # convo_state
        None,                                       # selected_scene_state
        gr.update(interactive=False),               # delete_btn
        [],                                         # chatbot
        NO_PROJECT_MD,                              # synopsis_md
        "",                                         # synopsis_box
        NO_PROJECT_MD,                              # style_md
        "",                                         # style_box
        None,                                       # style_image
        "",                                         # style_prompt_used
        gr.update(choices=[], value=None),          # char_picker
        NO_PROJECT_MD,                              # char_desc_md
        "",                                         # char_desc_box
        [],                                         # char_gallery
        [],                                         # scenes_table
        NO_PROJECT_MD,                              # scene_desc_md
        "",                                         # scene_desc_box
        None,                                       # scene_image
        "",                                         # scene_prompt_used
        gr.update(interactive=False),               # redo_btn
        gr.update(interactive=False),               # delete_version_btn
        gr.update(interactive=False),               # set_current_btn
        [],                                         # scene_versions_gallery
        "",                                         # selected_version_state
        "",                                         # synopsis_en_box
        "",                                         # style_en_box
        "",                                         # char_desc_en_box
        "",                                         # scene_prompt_en_box
    )


def _hydrate_outputs(project: Project, convo: StoryConversation):
    chars = list(project.characters.keys())
    first_char = chars[0] if chars else None
    first_scene_id = project.scenes[0].id if project.scenes else None

    char_desc = project.characters[first_char].description if first_char else ""
    char_refs = _ref_images_for(project, first_char)

    if first_scene_id is not None:
        s = project.scene_by_id(first_scene_id)
        scene_desc = s.prompt
        scene_img = _safe_path(s.image_path)
        scene_prompt = s.last_full_prompt or ""
        scene_prompt_en = s.prompt_en or ""
    else:
        scene_desc = ""
        scene_img = None
        scene_prompt = ""
        scene_prompt_en = ""

    redo_btn_u = _redo_btn_update(project, first_scene_id)
    del_ver_u, set_cur_u = _version_btn_updates(project, first_scene_id)

    return (
        project,
        convo,
        first_scene_id,
        gr.update(interactive=True),
        [],
        _md_or_empty(project.synopsis),
        project.synopsis or "",
        _md_or_empty(project.style_anchor),
        project.style_anchor or "",
        _safe_path(project.style_image),
        project.style_image_prompt or "",
        gr.update(choices=chars, value=first_char),
        _md_or_empty(char_desc) if first_char else "_no characters yet_",
        char_desc,
        char_refs,
        _scenes_rows(project, first_scene_id),
        _md_or_empty(scene_desc) if first_scene_id is not None else "_no scenes yet_",
        scene_desc,
        scene_img,
        scene_prompt,
        redo_btn_u,
        del_ver_u,
        set_cur_u,
        _scene_versions(project, first_scene_id),
        "",
        project.synopsis_en or "",
        project.style_anchor_en or "",
        (project.characters[first_char].description_en if first_char else "") or "",
        scene_prompt_en,
    )


# ---------- callbacks: chat ----------

def cb_chat(message, history, project, convo, selected_scene_id, char_name):
    if not message or not message.strip():
        yield (message, history, project, convo) + _refresh_views(project, selected_scene_id, char_name)
        return

    history = list(history or [])
    history.append({"role": "user", "content": message})

    # Show user msg + a pending assistant placeholder so Gradio renders its
    # built-in loading dots while we wait for the LLM.
    pending = {
        "role": "assistant",
        "content": "Thinking…",
        "metadata": {"status": "pending"},
    }
    yield ("", history + [pending], project, convo) + _refresh_views(project, selected_scene_id, char_name)

    if project is None or convo is None:
        history.append({"role": "assistant", "content": "Load or create a project first."})
        yield ("", history, project, convo) + _refresh_views(project, selected_scene_id, char_name)
        return

    try:
        reply, applied = convo.turn(message)
    except Exception as e:
        history.append({"role": "assistant", "content": f"[error] {e}"})
        yield ("", history, project, convo) + _refresh_views(project, selected_scene_id, char_name)
        return

    history.append({"role": "assistant", "content": reply})
    if applied:
        _save(project)
    yield ("", history, project, convo) + _refresh_views(project, selected_scene_id, char_name)


def _refresh_views(project: Project | None, selected_scene_id, char_name=None):
    if not project:
        return (
            NO_PROJECT_MD, "",
            NO_PROJECT_MD, "",
            gr.update(choices=[], value=None),
            NO_PROJECT_MD, "",
            [],
            [],
            NO_PROJECT_MD, "",
            None,
            "", "", "", "",
        )
    chars = list(project.characters.keys())
    # If the currently selected character no longer exists, pick the first one
    if char_name and char_name in project.characters:
        cur_char = char_name
    else:
        cur_char = chars[0] if chars else None

    if cur_char:
        char_desc = project.characters[cur_char].description
        char_desc_val = _md_or_empty(char_desc)
        char_desc_box_val = char_desc
        char_gallery_val = _ref_images_for(project, cur_char)
        char_desc_en_val = project.characters[cur_char].description_en or ""
    else:
        char_desc_val = "_no characters yet_"
        char_desc_box_val = ""
        char_gallery_val = []
        char_desc_en_val = ""

    if selected_scene_id is not None:
        s = project.scene_by_id(int(selected_scene_id))
        if s:
            scene_desc_val = _md_or_empty(s.prompt)
            scene_desc_box_val = s.prompt
            scene_img_val = _safe_path(s.image_path)
            scene_prompt_en_val = s.prompt_en or ""
        else:
            scene_desc_val = "_no scenes yet_"
            scene_desc_box_val = ""
            scene_img_val = None
            scene_prompt_en_val = ""
    else:
        scene_desc_val = "_no scenes yet_"
        scene_desc_box_val = ""
        scene_img_val = None
        scene_prompt_en_val = ""

    return (
        _md_or_empty(project.synopsis),
        project.synopsis or "",
        _md_or_empty(project.style_anchor),
        project.style_anchor or "",
        gr.update(choices=chars, value=cur_char),
        char_desc_val,
        char_desc_box_val,
        char_gallery_val,
        _scenes_rows(project, selected_scene_id),
        scene_desc_val,
        scene_desc_box_val,
        scene_img_val,
        project.synopsis_en or "",
        project.style_anchor_en or "",
        char_desc_en_val,
        scene_prompt_en_val,
    )


# ---------- callbacks: project tab edits ----------

def cb_save_synopsis(project, text):
    if not project:
        gr.Warning("No project loaded")
        return gr.update(), gr.update(), gr.update(), gr.update()
    project.synopsis = text or ""
    project.synopsis_en = translate_to_english(client, project.synopsis)
    _save(project)
    gr.Info("Synopsis saved")
    return (
        gr.update(value=_md_or_empty(project.synopsis), visible=True),
        gr.update(visible=False),
        gr.update(visible=False),
        project.synopsis_en,
    )


def cb_save_style(project, text):
    if not project:
        gr.Warning("No project loaded")
        return gr.update(), gr.update(), gr.update(), gr.update()
    project.style_anchor = text or ""
    project.style_anchor_en = translate_to_english(client, project.style_anchor)
    _save(project)
    gr.Info("Style anchor saved")
    return (
        gr.update(value=_md_or_empty(project.style_anchor), visible=True),
        gr.update(visible=False),
        gr.update(visible=False),
        project.style_anchor_en,
    )


# ---------- callbacks: characters tab ----------

def cb_select_character(project, name):
    if not project or not name or name not in project.characters:
        return NO_PROJECT_MD, "", [], ""
    c = project.characters[name]
    return _md_or_empty(c.description), c.description, _ref_images_for(project, name), c.description_en or ""


def cb_save_character(project, name, desc):
    if not project or not name or name not in project.characters:
        gr.Warning("No character selected")
        return gr.update(), gr.update(), gr.update(), gr.update()
    project.characters[name].description = desc or ""
    project.characters[name].description_en = translate_to_english(client, desc or "")
    _save(project)
    gr.Info(f"Saved {name}")
    return (
        gr.update(value=_md_or_empty(desc), visible=True),
        gr.update(visible=False),
        gr.update(visible=False),
        project.characters[name].description_en,
    )


# ---------- callbacks: scenes tab ----------

def cb_select_scene_row(project, current_id, evt: gr.SelectData):
    disabled = gr.update(interactive=False)
    if not project:
        return (current_id, gr.update(), NO_PROJECT_MD, "", None, "",
                [], "", disabled, disabled, disabled, "")
    idx = evt.index[0] if isinstance(evt.index, (list, tuple)) else evt.index
    if idx is None or idx < 0 or idx >= len(project.scenes):
        return (current_id, gr.update(), NO_PROJECT_MD, "", None, "",
                [], "", disabled, disabled, disabled, "")
    s = project.scenes[idx]
    del_ver_u, set_cur_u = _version_btn_updates(project, s.id)
    return (
        s.id,
        _scenes_rows(project, s.id),
        _md_or_empty(s.prompt),
        s.prompt,
        _safe_path(s.image_path),
        s.last_full_prompt or "",
        _scene_versions(project, s.id),
        "",
        _redo_btn_update(project, s.id),
        del_ver_u,
        set_cur_u,
        s.prompt_en or "",
    )


def cb_save_scene_desc(project, scene_id, desc):
    if not project or scene_id is None:
        gr.Warning("No scene selected")
        return gr.update(), gr.update(), gr.update(), _scenes_rows(project, scene_id), gr.update()
    s = project.scene_by_id(int(scene_id))
    if not s:
        gr.Warning("Scene not found")
        return gr.update(), gr.update(), gr.update(), _scenes_rows(project, scene_id), gr.update()
    s.prompt = desc or ""
    s.prompt_en = translate_to_english(client, s.prompt)
    _save(project)
    gr.Info(f"Saved scene {s.id}")
    return (
        gr.update(value=_md_or_empty(desc), visible=True),
        gr.update(visible=False),
        gr.update(visible=False),
        _scenes_rows(project, scene_id),
        s.prompt_en,
    )


def _gen_outputs(scene_image, project, scene_id, prompt, char_name):
    del_ver_u, set_cur_u = _version_btn_updates(project, scene_id)
    return (
        scene_image,
        _scenes_rows(project, scene_id),
        prompt,
        _scene_versions(project, scene_id),
        _ref_images_for(project, char_name),
        _redo_btn_update(project, scene_id),
        del_ver_u,
        set_cur_u,
    )


def cb_gen(project, scene_id, extra_fragment, char_name, new_seed: bool):
    if not project:
        gr.Warning("No project")
        return _gen_outputs(None, project, scene_id, "", char_name)
    if scene_id is None:
        gr.Warning("Pick a scene first")
        return _gen_outputs(None, project, scene_id, "", char_name)
    scene = project.scene_by_id(int(scene_id))
    if not scene:
        gr.Warning(f"Scene {scene_id} not found")
        return _gen_outputs(None, project, scene_id, "", char_name)
    if new_seed:
        scene.seed = None
    try:
        path, full_prompt = generate_scene_image(
            client, project, scene, _output_dir(project.name),
            extra_prompt_fragment=extra_fragment or "",
        )
        _save(project)
        gr.Info(f"Scene {scene.id} v{len(scene.versions) + 1} (seed={scene.seed})")
        return _gen_outputs(str(path), project, scene_id, full_prompt, char_name)
    except Exception as e:
        gr.Warning(f"Error: {e}")
        return _gen_outputs(_safe_path(scene.image_path), project, scene_id, "", char_name)


# ---------- callbacks: scene version history ----------

def cb_select_version(project, scene_id, evt: gr.SelectData):
    versions = _scene_versions(project, scene_id)
    if not versions or evt.index is None or evt.index >= len(versions):
        return ""
    return versions[evt.index]


def cb_open_img_delete(selected_path):
    """Prepare the delete-image confirmation modal: set preview, return a flag."""
    if not selected_path:
        gr.Warning("Click a version in the gallery first")
        return None, "no"
    return selected_path, "yes"


def _del_outputs(project, scene_id, image_path, sel_state, char_name):
    del_ver_u, set_cur_u = _version_btn_updates(project, scene_id)
    return (
        _scene_versions(project, scene_id),
        _safe_path(image_path),
        _scenes_rows(project, scene_id),
        sel_state,
        _ref_images_for(project, char_name),
        _redo_btn_update(project, scene_id),
        del_ver_u,
        set_cur_u,
    )


def cb_delete_version(project, scene_id, selected_path, char_name):
    if not project or scene_id is None or not selected_path:
        cur = None
        if project and scene_id is not None and project.scene_by_id(int(scene_id)):
            cur = project.scene_by_id(int(scene_id)).image_path
        return _del_outputs(project, scene_id, cur, "", char_name)
    scene = project.scene_by_id(int(scene_id))
    if not scene:
        gr.Warning("Scene not found")
        return _del_outputs(project, scene_id, None, "", char_name)

    deleted_was_current = selected_path == scene.image_path
    if deleted_was_current:
        scene.image_path = scene.versions.pop() if scene.versions else None
    elif selected_path in scene.versions:
        scene.versions.remove(selected_path)
    else:
        gr.Warning("Version not in this scene")
        return _del_outputs(project, scene_id, scene.image_path, "", char_name)

    try:
        Path(selected_path).unlink(missing_ok=True)
    except OSError:
        pass

    _save(project)
    gr.Info("Deleted (previous version promoted)" if deleted_was_current else "Deleted past version")
    return _del_outputs(project, scene_id, scene.image_path, "", char_name)


def _set_outputs(project, scene_id, image_path, char_name):
    del_ver_u, set_cur_u = _version_btn_updates(project, scene_id)
    return (
        _scene_versions(project, scene_id),
        _safe_path(image_path),
        _scenes_rows(project, scene_id),
        _ref_images_for(project, char_name),
        _redo_btn_update(project, scene_id),
        del_ver_u,
        set_cur_u,
    )


def cb_set_current_version(project, scene_id, selected_path, char_name):
    if not project or scene_id is None:
        gr.Warning("Pick a scene first")
        return _set_outputs(project, scene_id, None, char_name)
    scene = project.scene_by_id(int(scene_id))
    if not scene:
        gr.Warning("Scene not found")
        return _set_outputs(project, scene_id, None, char_name)
    if not selected_path:
        gr.Warning("Click a version in the gallery first")
        return _set_outputs(project, scene_id, scene.image_path, char_name)
    if selected_path == scene.image_path:
        gr.Info("Already current")
        return _set_outputs(project, scene_id, scene.image_path, char_name)
    if selected_path not in scene.versions:
        gr.Warning("Version not in this scene")
        return _set_outputs(project, scene_id, scene.image_path, char_name)
    if scene.image_path:
        scene.versions.append(scene.image_path)
    scene.versions.remove(selected_path)
    scene.image_path = selected_path
    _save(project)
    gr.Info(f"Promoted — scene {scene.id}")
    return _set_outputs(project, scene_id, scene.image_path, char_name)


# ---------- callbacks: style anchor ----------

def cb_gen_style(project):
    cur_prompt = project.style_image_prompt if project else ""
    cur_image = _safe_path(project.style_image) if project else None
    if not project:
        gr.Warning("No project")
        return cur_image, cur_prompt
    if not project.style_anchor:
        gr.Warning("Set a style first (chat or edit)")
        return cur_image, cur_prompt
    try:
        path, prompt = generate_style_anchor_image(client, project, _output_dir(project.name))
        _save(project)
        gr.Info("Style anchor saved")
        return str(path), prompt
    except Exception as e:
        gr.Warning(f"Error: {e}")
        return cur_image, cur_prompt


# ---------- UI ----------

CSS = """
/* lock page to viewport — only the inner cols scroll */
html, body { margin: 0; height: 100vh; overflow: hidden; }
.gradio-container {
  max-width: 100% !important;
  height: 100vh !important;
  max-height: 100vh !important;
  overflow: hidden !important;
  padding: 0.6rem 1rem !important;
  box-sizing: border-box !important;
}

#topbar { align-items: center; gap: 0.4rem; }
.topbar-label { flex: 0 0 auto !important; min-width: 0 !important; }
.topbar-label, .topbar-label p { margin: 0 !important; padding: 0 0.2rem 0 0 !important; white-space: nowrap; }

/* both columns: same height = viewport minus chrome (title+topbar+footer+padding) */
#chat-col, #side-col {
  height: calc(100vh - 268px) !important;
  max-height: calc(100vh - 268px) !important;
  overflow-y: auto !important;
  padding-right: 0.4rem;
  scrollbar-width: none;
}
#chat-col::-webkit-scrollbar, #side-col::-webkit-scrollbar { width: 0; height: 0; display: none; }

/* lock state during LLM ops (chat stays unlocked) */
#side-col.ui-busy { pointer-events: none; opacity: 0.55; filter: grayscale(0.2); }
#side-col.ui-busy::after {
  content: "Working…"; position: sticky; top: 0;
  display: block; text-align: center; padding: 0.4rem;
  background: var(--color-accent, #f59e0b); color: #000; font-weight: 600;
  z-index: 5; pointer-events: none;
}

#title-bar { margin: 0 0 0.2rem 0; }
.icon-btn { min-width: 38px !important; max-width: 44px !important; padding: 6px !important; font-size: 1.05rem !important; }
.editable-row { align-items: center; gap: 0.4rem; }
.field-card { padding: 0.6rem 0.8rem; border-radius: 8px; background: var(--block-background-fill); border: 1px solid var(--border-color-primary); }
.dim, .dim p { opacity: 0.7; font-size: 0.85rem; }

/* destructive action palette (red) */
.btn-danger button, button.btn-danger {
  background: #dc2626 !important;
  border-color: #dc2626 !important;
  color: #fff !important;
}
.btn-danger button:hover, button.btn-danger:hover {
  background: #b91c1c !important;
  border-color: #b91c1c !important;
}
.btn-danger button:disabled, button.btn-danger:disabled,
.btn-danger button[disabled], button.btn-danger[disabled] {
  background: #5b1f1f !important;
  border-color: #5b1f1f !important;
  color: rgba(255, 255, 255, 0.55) !important;
  filter: none !important;
  opacity: 1 !important;
}

/* Gradio defaults `disabled = hover color` for primary/secondary, so disabled
   primary buttons turn brighter orange and secondary picks up the theme tint.
   Override per variant so each family keeps its identity even when disabled. */
button.primary[disabled], button.primary:disabled {
  background: #7a3a08 !important;
  border-color: #7a3a08 !important;
  color: rgba(255, 255, 255, 0.6) !important;
  filter: none !important;
  opacity: 1 !important;
}
button.secondary[disabled], button.secondary:disabled {
  background: #2c2c2c !important;
  border-color: #2c2c2c !important;
  color: rgba(255, 255, 255, 0.4) !important;
  filter: none !important;
  opacity: 1 !important;
}

/* ------ modals: separate fixed backdrop + fixed card, hidden via class ------ */
.modal-backdrop {
  position: fixed; inset: 0;
  background: rgba(0, 0, 0, 0.55);
  z-index: 9998;
}
.modal-card {
  position: fixed !important;
  top: 50% !important; left: 50% !important;
  transform: translate(-50%, -50%) !important;
  z-index: 9999 !important;
  background: var(--background-fill-primary, #1f1f1f) !important;
  padding: 1.4rem 1.5rem !important;
  border-radius: 12px !important;
  min-width: 460px !important; max-width: 640px !important;
  box-shadow: 0 25px 70px rgba(0, 0, 0, 0.55) !important;
  border: 1px solid var(--border-color-primary) !important;
}
.modal-hidden { display: none !important; }
.modal-card .modal-title { margin: 0 0 0.6rem 0; }

/* selected scene row marker + cursor hint */
.scenes-table table td:first-child { width: 26px; text-align: center; color: var(--color-accent, #f59e0b); font-weight: 700; }
.scenes-table table tbody tr { cursor: pointer; }

/* keep an element clickable but off-screen (used for the hidden send button) */
.offscreen {
  position: absolute !important;
  left: -9999px !important;
  width: 1px !important; height: 1px !important;
  overflow: hidden !important;
}
"""


# JS helpers — toggle .modal-hidden via class on backdrop + card.
JS_LOCK = """(...args) => {
  const el = document.getElementById('side-col');
  if (el) { el.classList.add('ui-busy'); }
  return args;
}"""
JS_UNLOCK = """() => {
  const el = document.getElementById('side-col');
  if (el) { el.classList.remove('ui-busy'); }
}"""

JS_OPEN_NEW = """() => {
  document.getElementById('modal-backdrop')?.classList.remove('modal-hidden');
  document.getElementById('new-modal')?.classList.remove('modal-hidden');
  document.getElementById('delete-modal')?.classList.add('modal-hidden');
  document.getElementById('img-delete-modal')?.classList.add('modal-hidden');
}"""
JS_CLOSE_NEW = """() => {
  document.getElementById('new-modal')?.classList.add('modal-hidden');
  if (!document.querySelector('.modal-card:not(.modal-hidden)')) {
    document.getElementById('modal-backdrop')?.classList.add('modal-hidden');
  }
}"""
JS_OPEN_DELETE = """() => {
  document.getElementById('modal-backdrop')?.classList.remove('modal-hidden');
  document.getElementById('delete-modal')?.classList.remove('modal-hidden');
  document.getElementById('new-modal')?.classList.add('modal-hidden');
  document.getElementById('img-delete-modal')?.classList.add('modal-hidden');
}"""
JS_CLOSE_DELETE = """() => {
  document.getElementById('delete-modal')?.classList.add('modal-hidden');
  if (!document.querySelector('.modal-card:not(.modal-hidden)')) {
    document.getElementById('modal-backdrop')?.classList.add('modal-hidden');
  }
}"""
JS_OPEN_IMG_DELETE = """(flag) => {
  if (flag === 'yes') {
    document.getElementById('modal-backdrop')?.classList.remove('modal-hidden');
    document.getElementById('img-delete-modal')?.classList.remove('modal-hidden');
  }
}"""
JS_CLOSE_IMG_DELETE = """() => {
  document.getElementById('img-delete-modal')?.classList.add('modal-hidden');
  if (!document.querySelector('.modal-card:not(.modal-hidden)')) {
    document.getElementById('modal-backdrop')?.classList.add('modal-hidden');
  }
}"""

# Swap Enter / Shift+Enter behaviour on the chat textarea.
# Gradio Textbox internals listen on **keypress** (not keydown), and submit
# fires on Shift+Enter when lines>1. So:
# - Real Enter alone:    prevent default, dispatch a synthetic *keypress*
#                        with shiftKey=true on the textarea — Gradio's keypress
#                        handler then submits with the current textbox value.
# - Real Shift+Enter:    prevent default (which also suppresses the keypress so
#                        Gradio doesn't submit) and insert a newline ourselves.
JS_BIND_ENTER_SUBMIT = """() => {
  if (window._enterSubmitBound) return;
  window._enterSubmitBound = true;
  document.addEventListener('keydown', (e) => {
    if (e.key !== 'Enter' || e.ctrlKey || e.metaKey || e.isComposing) return;
    const ta = e.target;
    if (!ta || ta.tagName !== 'TEXTAREA') return;
    if (!ta.closest('#chat-col')) return;
    if (e.shiftKey) {
      // user pressed Shift+Enter -> newline; suppress Gradio's submit
      e.preventDefault();
      e.stopImmediatePropagation();
      const s = ta.selectionStart, en = ta.selectionEnd;
      ta.value = ta.value.slice(0, s) + '\\n' + ta.value.slice(en);
      ta.selectionStart = ta.selectionEnd = s + 1;
      ta.dispatchEvent(new Event('input', { bubbles: true }));
    } else {
      // user pressed Enter -> trigger submit by emitting a synthetic
      // shift-enter keypress (the format Gradio listens for).
      e.preventDefault();
      e.stopImmediatePropagation();
      ta.dispatchEvent(new KeyboardEvent('keypress', {
        key: 'Enter', code: 'Enter', keyCode: 13, which: 13, charCode: 13,
        shiftKey: true, bubbles: true, cancelable: true,
      }));
    }
  }, true);
}"""


def _editable_block(label: str, lines: int = 4):
    with gr.Group(elem_classes="field-card"):
        with gr.Row(elem_classes="editable-row"):
            md = gr.Markdown(NO_PROJECT_MD)
            edit_btn = gr.Button(PENCIL, scale=0, elem_classes="icon-btn", min_width=44)
        box = gr.Textbox(label=label, lines=lines, visible=False, show_label=False, container=False)
        with gr.Row(visible=False) as actions:
            save_btn = gr.Button("Save", variant="primary", scale=0)
            cancel_btn = gr.Button("Cancel", scale=0)
    return md, edit_btn, box, actions, save_btn, cancel_btn


def build_ui():
    with gr.Blocks(title="NVIDIA Illustration Agent", fill_height=True) as demo:
        project_state = gr.State(None)
        convo_state = gr.State(None)
        selected_scene_state = gr.State(None)
        selected_version_state = gr.State("")
        # Tiny relay so JS knows whether to actually open the image-delete modal
        # (skip if no version is currently selected).
        img_delete_open_flag = gr.Textbox(visible=False)

        with gr.Column(elem_id="app-shell"):
            with gr.Row(elem_id="title-bar"):
                gr.Markdown("## NVIDIA Illustration Agent")

            with gr.Row(elem_id="topbar"):
                gr.Markdown("**Project**", elem_classes=["topbar-label"])
                project_picker = gr.Dropdown(
                    show_label=False,
                    container=False,
                    choices=_list_projects(),
                    value=None,
                    allow_custom_value=False,
                    scale=6,
                )
                new_btn = gr.Button("+ New", scale=0, min_width=90)
                delete_btn = gr.Button(
                    "Delete", scale=0, min_width=90,
                    elem_classes=["btn-danger"], interactive=False,
                )

            with gr.Row(elem_id="main-row"):
                with gr.Column(scale=4, elem_id="chat-col"):
                    gr.Markdown("### Story chat")
                    chatbot = gr.Chatbot(
                        label="Story co-writer", height="53vh", show_label=False,
                    )
                    msg_in = gr.Textbox(
                        placeholder="Talk about your story… say 'lock it' to commit. (Enter to send, Shift+Enter for new line)",
                        show_label=False, container=False,
                        lines=3, max_lines=3,
                        elem_id="chat-input",
                    )
                    # Hidden send button: stays mounted (clickable) but visually
                    # offscreen. Triggered by Enter-key JS handler on the textarea.
                    send_btn = gr.Button(
                        "Send", elem_id="chat-send", elem_classes=["offscreen"],
                    )

                with gr.Column(scale=6, elem_id="side-col"):
                    with gr.Tabs():
                        # ---- PROJECT TAB ----
                        with gr.Tab("Project"):
                            gr.Markdown("#### Synopsis")
                            (synopsis_md, synopsis_edit_btn, synopsis_box,
                             synopsis_actions, synopsis_save_btn, synopsis_cancel_btn) = _editable_block(
                                "Synopsis", lines=5
                            )
                            synopsis_en_box = gr.Textbox(
                                label="English version (for image generation)",
                                interactive=False, lines=2, show_label=True,
                            )

                            gr.Markdown("#### Style anchor (text)")
                            (style_md, style_edit_btn, style_box,
                             style_actions, style_save_btn, style_cancel_btn) = _editable_block(
                                "Style anchor", lines=4
                            )
                            style_en_box = gr.Textbox(
                                label="English version (for image generation)",
                                interactive=False, lines=2, show_label=True,
                            )

                            with gr.Accordion("Style anchor image", open=False):
                                gr.Markdown(
                                    "Generate a single visual reference once your style "
                                    "tags are locked. The prompt is built ONLY from the "
                                    "style anchor text above (translated to English).",
                                    elem_classes=["dim"],
                                )
                                gen_style_btn = gr.Button(
                                    "Generate style anchor", variant="primary"
                                )
                                style_image = gr.Image(show_label=False, height=420)
                                style_prompt_used = gr.Textbox(
                                    label="Final prompt sent to FLUX (style anchor)",
                                    interactive=False,
                                    lines=4,
                                )

                        # ---- CHARACTERS TAB ----
                        with gr.Tab("Characters"):
                            char_picker = gr.Dropdown(
                                label="Character", choices=[], value=None
                            )
                            gr.Markdown("#### Visual description")
                            (char_desc_md, char_desc_edit_btn, char_desc_box,
                             char_desc_actions, char_desc_save_btn, char_desc_cancel_btn) = _editable_block(
                                "Visual description (used in every prompt)", lines=6
                            )
                            char_desc_en_box = gr.Textbox(
                                label="English version (for image generation)",
                                interactive=False, lines=3, show_label=True,
                            )
                            gr.Markdown("#### Past appearances")
                            char_gallery = gr.Gallery(
                                show_label=False, columns=4, height=260,
                            )

                        # ---- SCENES TAB ----
                        with gr.Tab("Scenes"):
                            scenes_table = gr.Dataframe(
                                headers=["", "ID", "Title", "Characters", "Image"],
                                datatype=["str", "number", "str", "str", "str"],
                                interactive=False, wrap=True, show_label=False,
                                elem_classes=["scenes-table"],
                            )
                            gr.Markdown("#### Scene description")
                            (scene_desc_md, scene_desc_edit_btn, scene_desc_box,
                             scene_desc_actions, scene_desc_save_btn, scene_desc_cancel_btn) = _editable_block(
                                "Scene description (used directly in prompt)", lines=4
                            )
                            scene_prompt_en_box = gr.Textbox(
                                label="English version (for image generation)",
                                interactive=False, lines=3, show_label=True,
                            )

                            scene_image = gr.Image(label="Current image", height=420)
                            with gr.Row():
                                gen_btn = gr.Button("Generate", variant="primary")
                                redo_btn = gr.Button("Redo (new seed)", interactive=False)
                            with gr.Accordion("Advanced", open=False):
                                extra_fragment = gr.Textbox(
                                    label="Extra prompt fragment (optional)",
                                    placeholder="e.g. paste a fix fragment from review",
                                )
                                scene_prompt_used = gr.Textbox(
                                    label="Final prompt sent to FLUX (last gen)",
                                    interactive=False, lines=4,
                                )
                            with gr.Accordion("Version history", open=False):
                                gr.Markdown(
                                    "Click a thumbnail to select, then delete or promote it.",
                                    elem_classes=["dim"],
                                )
                                scene_versions_gallery = gr.Gallery(
                                    show_label=False, columns=4, height=240, allow_preview=True,
                                )
                                with gr.Row():
                                    delete_version_btn = gr.Button(
                                        "Delete selected version",
                                        elem_classes=["btn-danger"],
                                        interactive=False,
                                    )
                                    set_current_btn = gr.Button(
                                        "Set selected as current", interactive=False,
                                    )

        # ---- modals: backdrop + cards (siblings, hidden via class) ----
        gr.HTML('<div id="modal-backdrop" class="modal-backdrop modal-hidden"></div>')

        with gr.Column(elem_id="new-modal", elem_classes=["modal-card", "modal-hidden"]):
            gr.Markdown("### New project", elem_classes=["modal-title"])
            new_name_box = gr.Textbox(
                label="Project name", placeholder="my_story", autofocus=True,
            )
            with gr.Row():
                new_create_btn = gr.Button("Create", variant="primary")
                new_cancel_btn = gr.Button("Cancel")

        with gr.Column(elem_id="delete-modal", elem_classes=["modal-card", "modal-hidden"]):
            gr.Markdown("### Delete project", elem_classes=["modal-title"])
            gr.Markdown(
                "_This will permanently remove the project and all its images._",
                elem_classes=["dim"],
            )
            delete_confirm_box = gr.Textbox(
                label="Type the project name to confirm",
                placeholder="exact project name",
            )
            with gr.Row():
                delete_yes_btn = gr.Button("Confirm delete", elem_classes=["btn-danger"])
                delete_no_btn = gr.Button("Cancel")

        with gr.Column(elem_id="img-delete-modal", elem_classes=["modal-card", "modal-hidden"]):
            gr.Markdown("### Delete this image?", elem_classes=["modal-title"])
            img_delete_preview = gr.Image(
                show_label=False, height=320, interactive=False,
            )
            with gr.Row():
                img_delete_yes_btn = gr.Button("Delete", elem_classes=["btn-danger"])
                img_delete_no_btn = gr.Button("Cancel")

        # ============ wiring ============

        load_outputs = [
            project_state,
            convo_state,
            selected_scene_state,
            delete_btn,
            chatbot,
            synopsis_md,
            synopsis_box,
            style_md,
            style_box,
            style_image,
            style_prompt_used,
            char_picker,
            char_desc_md,
            char_desc_box,
            char_gallery,
            scenes_table,
            scene_desc_md,
            scene_desc_box,
            scene_image,
            scene_prompt_used,
            redo_btn,
            delete_version_btn,
            set_current_btn,
            scene_versions_gallery,
            selected_version_state,
            synopsis_en_box,
            style_en_box,
            char_desc_en_box,
            scene_prompt_en_box,
        ]

        project_picker.change(cb_load, [project_picker], load_outputs)

        # NEW project flow
        new_btn.click(fn=None, js=JS_OPEN_NEW)
        new_cancel_btn.click(
            lambda: gr.update(value=""), None, [new_name_box], js=JS_CLOSE_NEW,
        )
        new_create_btn.click(
            cb_create, [new_name_box],
            load_outputs + [project_picker, new_name_box],
        ).then(fn=None, js=JS_CLOSE_NEW)
        new_name_box.submit(
            cb_create, [new_name_box],
            load_outputs + [project_picker, new_name_box],
        ).then(fn=None, js=JS_CLOSE_NEW)

        # DELETE project flow
        delete_btn.click(fn=None, js=JS_OPEN_DELETE)
        delete_no_btn.click(
            lambda: gr.update(value=""), None, [delete_confirm_box], js=JS_CLOSE_DELETE,
        )
        delete_yes_btn.click(
            cb_delete_confirm, [project_state, delete_confirm_box],
            load_outputs + [project_picker, delete_confirm_box],
        ).then(fn=None, js=JS_CLOSE_DELETE)

        # CHAT
        chat_refresh_outputs = [
            synopsis_md, synopsis_box,
            style_md, style_box,
            char_picker,
            char_desc_md, char_desc_box,
            char_gallery,
            scenes_table,
            scene_desc_md, scene_desc_box,
            scene_image,
            synopsis_en_box,
            style_en_box,
            char_desc_en_box,
            scene_prompt_en_box,
        ]
        chat_outputs = [msg_in, chatbot, project_state, convo_state] + chat_refresh_outputs
        # Both wired: hidden send_btn fires from JS on Enter; msg_in.submit is a
        # safety net for single-line submit triggers.
        send_btn.click(
            cb_chat,
            [msg_in, chatbot, project_state, convo_state, selected_scene_state, char_picker],
            chat_outputs,
        )
        msg_in.submit(
            cb_chat,
            [msg_in, chatbot, project_state, convo_state, selected_scene_state, char_picker],
            chat_outputs,
        )

        # ---- inline edit wiring ----
        synopsis_edit_btn.click(_show_edit, None, [synopsis_md, synopsis_box, synopsis_actions])
        synopsis_cancel_btn.click(_hide_edit, None, [synopsis_md, synopsis_box, synopsis_actions])
        synopsis_save_btn.click(
            cb_save_synopsis, [project_state, synopsis_box],
            [synopsis_md, synopsis_box, synopsis_actions, synopsis_en_box],
        )
        style_edit_btn.click(_show_edit, None, [style_md, style_box, style_actions])
        style_cancel_btn.click(_hide_edit, None, [style_md, style_box, style_actions])
        style_save_btn.click(
            cb_save_style, [project_state, style_box],
            [style_md, style_box, style_actions, style_en_box],
        )

        # ---- characters ----
        char_picker.change(
            cb_select_character, [project_state, char_picker],
            [char_desc_md, char_desc_box, char_gallery, char_desc_en_box],
        )
        char_desc_edit_btn.click(_show_edit, None, [char_desc_md, char_desc_box, char_desc_actions])
        char_desc_cancel_btn.click(_hide_edit, None, [char_desc_md, char_desc_box, char_desc_actions])
        char_desc_save_btn.click(
            cb_save_character, [project_state, char_picker, char_desc_box],
            [char_desc_md, char_desc_box, char_desc_actions, char_desc_en_box],
        )

        # ---- scenes ----
        scenes_table.select(
            cb_select_scene_row,
            [project_state, selected_scene_state],
            [
                selected_scene_state,
                scenes_table,
                scene_desc_md,
                scene_desc_box,
                scene_image,
                scene_prompt_used,
                scene_versions_gallery,
                selected_version_state,
                redo_btn,
                delete_version_btn,
                set_current_btn,
                scene_prompt_en_box,
            ],
        )
        scene_desc_edit_btn.click(_show_edit, None, [scene_desc_md, scene_desc_box, scene_desc_actions])
        scene_desc_cancel_btn.click(_hide_edit, None, [scene_desc_md, scene_desc_box, scene_desc_actions])
        scene_desc_save_btn.click(
            cb_save_scene_desc,
            [project_state, selected_scene_state, scene_desc_box],
            [scene_desc_md, scene_desc_box, scene_desc_actions, scenes_table, scene_prompt_en_box],
        )

        gen_outputs = [
            scene_image, scenes_table, scene_prompt_used,
            scene_versions_gallery, char_gallery, redo_btn,
            delete_version_btn, set_current_btn,
        ]
        gen_btn.click(
            lambda p, sid, frag, cn: cb_gen(p, sid, frag, cn, new_seed=False),
            [project_state, selected_scene_state, extra_fragment, char_picker],
            gen_outputs, js=JS_LOCK,
        ).then(fn=None, js=JS_UNLOCK)
        redo_btn.click(
            lambda p, sid, frag, cn: cb_gen(p, sid, frag, cn, new_seed=True),
            [project_state, selected_scene_state, extra_fragment, char_picker],
            gen_outputs, js=JS_LOCK,
        ).then(fn=None, js=JS_UNLOCK)

        scene_versions_gallery.select(
            cb_select_version,
            [project_state, selected_scene_state],
            [selected_version_state],
        )

        # delete-version: open modal with preview instead of deleting directly
        delete_version_btn.click(
            cb_open_img_delete,
            [selected_version_state],
            [img_delete_preview, img_delete_open_flag],
        ).then(fn=None, inputs=[img_delete_open_flag], js=JS_OPEN_IMG_DELETE)

        img_delete_no_btn.click(fn=None, js=JS_CLOSE_IMG_DELETE)
        img_delete_yes_btn.click(
            cb_delete_version,
            [project_state, selected_scene_state, selected_version_state, char_picker],
            [
                scene_versions_gallery, scene_image, scenes_table,
                selected_version_state, char_gallery, redo_btn,
                delete_version_btn, set_current_btn,
            ],
        ).then(fn=None, js=JS_CLOSE_IMG_DELETE)

        set_current_btn.click(
            cb_set_current_version,
            [project_state, selected_scene_state, selected_version_state, char_picker],
            [
                scene_versions_gallery, scene_image, scenes_table,
                char_gallery, redo_btn, delete_version_btn, set_current_btn,
            ],
        )

        gen_style_btn.click(
            cb_gen_style, [project_state], [style_image, style_prompt_used],
            js=JS_LOCK,
        ).then(fn=None, js=JS_UNLOCK)

        # Enter = submit, Shift+Enter = newline. Bind once when the textarea mounts.
        demo.load(fn=None, js=JS_BIND_ENTER_SUBMIT)

    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.queue()
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        inbrowser=True,
        theme=gr.themes.Soft(primary_hue="orange"),
        css=CSS,
        allowed_paths=[str(OUTPUT_BASE)],
    )
