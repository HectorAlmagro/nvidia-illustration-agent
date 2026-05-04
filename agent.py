from __future__ import annotations
import sys
import argparse
from pathlib import Path
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.markdown import Markdown

from nvidia_client import NvidiaClient
from state import Project
from story import StoryConversation
from image_gen import generate_scene_image, generate_style_anchor_image


HELP = """
Comandos:
  /help                 Show this help
  /show                 Show project state (synopsis, characters, scenes)
  /scenes               List scenes
  /chars                List characters
  /style                Generate the style-anchor image (run once after locking style)
  /gen <scene_id>       Generate (or regenerate) image for a scene
  /redo <scene_id>      Regenerate image for a scene with new seed
  /open <scene_id>      Open the current image of a scene (macOS `open`)
  /save                 Save project to disk
  /load <path>          Load project from disk
  /quit                 Exit

Anything else = chat with the story co-writer LLM.
Confirm with words like "ok", "lock it", "go" to make it emit the JSON
project state, which is auto-applied.
""".strip()


def cmd_show(console: Console, project: Project) -> None:
    console.print(Panel.fit(project.synopsis or "(no synopsis)", title="Synopsis"))
    console.print(Panel.fit(project.style_anchor or "(no style)", title="Style"))
    if project.characters:
        t = Table(title="Characters")
        t.add_column("Name")
        t.add_column("Description")
        for c in project.characters.values():
            t.add_row(c.name, c.description[:120])
        console.print(t)
    if project.scenes:
        t = Table(title="Scenes")
        t.add_column("ID")
        t.add_column("Title")
        t.add_column("Chars")
        t.add_column("Img?")
        for s in project.scenes:
            t.add_row(
                str(s.id),
                s.title,
                ",".join(s.characters),
                "yes" if s.image_path else "no",
            )
        console.print(t)


def cmd_gen(
    console: Console,
    client: NvidiaClient,
    project: Project,
    output_dir: Path,
    sid: int,
    *,
    new_seed: bool = False,
) -> None:
    scene = project.scene_by_id(sid)
    if not scene:
        console.print(f"[red]No scene with id {sid}[/red]")
        return
    if new_seed:
        scene.seed = None
    console.print(f"[cyan]Generating scene {sid}: {scene.title}…[/cyan]")
    try:
        path, full_prompt = generate_scene_image(client, project, scene, output_dir)
        console.print(f"[green]Saved:[/green] {path}")
        console.print(f"[dim]Prompt:[/dim] {full_prompt}")
    except Exception as e:
        console.print(f"[red]Image gen failed:[/red] {e}")


def cmd_style(
    console: Console, client: NvidiaClient, project: Project, output_dir: Path
) -> None:
    if not project.style_anchor:
        console.print("[yellow]No style_anchor defined. Lock the story first.[/yellow]")
        return
    console.print("[cyan]Generating style anchor…[/cyan]")
    try:
        path = generate_style_anchor_image(client, project, output_dir)
        console.print(f"[green]Saved:[/green] {path}")
    except Exception as e:
        console.print(f"[red]Style anchor failed:[/red] {e}")


def main() -> int:
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", default="my_story", help="Project name")
    parser.add_argument(
        "--load",
        type=str,
        default=None,
        help="Path to existing project JSON to load",
    )
    args = parser.parse_args()

    console = Console()
    client = NvidiaClient()

    base = Path(__file__).parent
    projects_dir = base / "projects"
    projects_dir.mkdir(exist_ok=True)

    if args.load:
        project = Project.load(Path(args.load))
        console.print(f"[green]Loaded project:[/green] {project.name}")
    else:
        project_path = projects_dir / f"{args.project}.json"
        if project_path.exists():
            project = Project.load(project_path)
            console.print(f"[green]Loaded existing:[/green] {project_path}")
        else:
            project = Project(name=args.project)
            console.print(f"[cyan]New project:[/cyan] {project.name}")

    output_dir = base / "output" / project.name
    output_dir.mkdir(parents=True, exist_ok=True)
    save_path = projects_dir / f"{project.name}.json"

    convo = StoryConversation(client, project)

    console.print(Panel.fit(HELP, title="Illustration Agent"))

    while True:
        try:
            line = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[cyan]Saving and exiting…[/cyan]")
            project.save(save_path)
            return 0
        if not line:
            continue

        if line.startswith("/"):
            parts = line.split()
            cmd = parts[0].lower()
            arg = parts[1] if len(parts) > 1 else None

            if cmd == "/help":
                console.print(Panel.fit(HELP, title="Help"))
            elif cmd == "/show":
                cmd_show(console, project)
            elif cmd == "/scenes":
                for s in project.scenes:
                    console.print(
                        f"  [{s.id:02d}] {s.title} "
                        f"({','.join(s.characters)}) "
                        f"img={'yes' if s.image_path else 'no'}"
                    )
            elif cmd == "/chars":
                for c in project.characters.values():
                    console.print(f"  - {c.name}: {c.description[:200]}")
            elif cmd == "/style":
                cmd_style(console, client, project, output_dir)
                project.save(save_path)
            elif cmd == "/gen":
                if arg is None:
                    console.print("[red]Usage: /gen <scene_id>[/red]")
                else:
                    cmd_gen(console, client, project, output_dir, int(arg))
                    project.save(save_path)
            elif cmd == "/redo":
                if arg is None:
                    console.print("[red]Usage: /redo <scene_id>[/red]")
                else:
                    cmd_gen(
                        console,
                        client,
                        project,
                        output_dir,
                        int(arg),
                        new_seed=True,
                    )
                    project.save(save_path)
            elif cmd == "/open":
                if arg is None:
                    console.print("[red]Usage: /open <scene_id>[/red]")
                else:
                    s = project.scene_by_id(int(arg))
                    if s and s.image_path:
                        import subprocess

                        subprocess.run(["open", s.image_path], check=False)
                    else:
                        console.print("[yellow]No image[/yellow]")
            elif cmd == "/save":
                project.save(save_path)
                console.print(f"[green]Saved:[/green] {save_path}")
            elif cmd == "/load":
                if arg is None:
                    console.print("[red]Usage: /load <path>[/red]")
                else:
                    project = Project.load(Path(arg))
                    convo = StoryConversation(client, project)
                    save_path = projects_dir / f"{project.name}.json"
                    output_dir = base / "output" / project.name
                    output_dir.mkdir(parents=True, exist_ok=True)
                    console.print(f"[green]Loaded:[/green] {project.name}")
            elif cmd in ("/quit", "/exit"):
                project.save(save_path)
                console.print(f"[green]Saved:[/green] {save_path}")
                return 0
            else:
                console.print(f"[red]Unknown command:[/red] {cmd}")
            continue

        try:
            reply, applied = convo.turn(line)
        except Exception as e:
            console.print(f"[red]LLM error:[/red] {e}")
            continue
        console.print(Markdown(reply))
        if applied:
            console.print(
                "[green]Story state updated.[/green] "
                f"{len(project.characters)} chars, {len(project.scenes)} scenes."
            )
            project.save(save_path)


if __name__ == "__main__":
    sys.exit(main())
