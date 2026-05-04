# nvidia-illustration-agent

An AI-powered story co-writer and illustration agent. Chat with an LLM to develop a short illustrated story — characters, scenes, and visual style — then generate images for each scene using NVIDIA's image generation models.

---

## Features

- **Story co-writer**: Chat-based workflow to build a story (synopsis, characters, scenes, visual style).
- **Image generation**: Generates a style-anchor image and per-scene illustrations using FLUX models via NVIDIA AI APIs.
- **Two interfaces**:
  - **Gradio web UI** (`ui.py`) — browser-based chat and image viewer.
  - **CLI** (`agent.py`) — terminal-based interactive session.
- **Project persistence**: Projects are saved as JSON files under `projects/` and can be reloaded at any time.
- **Output management**: Generated images are saved under `output/<project-name>/`.

---

## Requirements

- Python 3.10+
- An [NVIDIA AI API key](https://build.nvidia.com/) (free tier available)

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/your-username/nvidia-illustration-agent.git
cd nvidia-illustration-agent
```

### 2. Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip3 install -r requirements.txt
```

### 4. Configure your API key

Create a `.env` file in the root of the project:

```bash
cp .env.example .env   # if an example exists, otherwise create it manually
```

Or create it directly:

```bash
echo "NVIDIA_API_KEY=your_api_key_here" > .env
```

Your `.env` file should look like this:

```env
NVIDIA_API_KEY=nvapi-xxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

You can get your API key from [https://build.nvidia.com/](https://build.nvidia.com/).

---

## Running the app

### Web UI (recommended)

```bash
python3 ui.py
```

Open your browser at [http://localhost:7860](http://localhost:7860).

### CLI

```bash
python3 agent.py
```

Available CLI commands once running:

| Command | Description |
|---|---|
| `/help` | Show all commands |
| `/show` | Display current project state |
| `/scenes` | List all scenes |
| `/chars` | List all characters |
| `/style` | Generate the style-anchor image |
| `/gen <scene_id>` | Generate image for a scene |
| `/redo <scene_id>` | Regenerate image with a new seed |
| `/open <scene_id>` | Open the scene image (macOS) |
| `/save` | Save project to disk |
| `/load <path>` | Load a project from disk |
| `/quit` | Exit |

Anything else typed in the CLI is sent to the story co-writer LLM. When ready to commit the story structure, confirm with phrases like **"ok"**, **"lock it"**, or **"go"** — the LLM will emit a JSON state that is automatically applied to the project.

---

## Project structure

```
nvidia-illustration-agent/
├── agent.py          # CLI entry point
├── ui.py             # Gradio web UI entry point
├── nvidia_client.py  # NVIDIA API client (LLM + image generation)
├── story.py          # Story co-writer conversation logic
├── image_gen.py      # Image generation helpers
├── state.py          # Project data model (Project, Character, Scene)
├── requirements.txt  # Python dependencies
├── .env              # API keys (not committed)
├── projects/         # Saved project JSON files
└── output/           # Generated images, organized by project name
```

---

## Models used

| Purpose | Model |
|---|---|
| Story co-writer (LLM) | `meta/llama-3.3-70b-instruct` |
| Vision (VLM) | `meta/llama-3.2-90b-vision-instruct` |
| Image generation | `black-forest-labs/flux.1-dev` |
| Image editing (Kontext) | `black-forest-labs/flux.1-kontext-dev` |

Models are configurable in `nvidia_client.py`.

---

## Supported image aspect ratios

| Ratio | Dimensions |
|---|---|
| 1:1 | 1024×1024 |
| 16:9 | 1344×768 |
| 9:16 | 768×1344 |
| 4:3 | 1152×896 |
| 3:4 | 896×1152 |
| 3:2 | 1216×832 |
| 2:3 | 832×1216 |
