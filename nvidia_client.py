from __future__ import annotations
import os
import base64
from pathlib import Path
from typing import Optional
import requests
from openai import OpenAI


LLM_BASE_URL = "https://integrate.api.nvidia.com/v1"
IMAGE_BASE_URL = "https://ai.api.nvidia.com/v1/genai"
NVCF_ASSETS_URL = "https://api.nvcf.nvidia.com/v2/nvcf/assets"

DEFAULT_LLM = "meta/llama-3.3-70b-instruct"
DEFAULT_VLM = "meta/llama-3.2-90b-vision-instruct"
DEFAULT_IMAGE_MODEL = "black-forest-labs/flux.1-dev"
DEFAULT_KONTEXT_MODEL = "black-forest-labs/flux.1-kontext-dev"

# FLUX.1-dev only accepts these dimensions.
FLUX_VALID_DIMS = (768, 832, 896, 960, 1024, 1088, 1152, 1216, 1280, 1344)


def _b64_image(path: str | Path) -> str:
    return base64.b64encode(Path(path).read_bytes()).decode()


def aspect_ratio_to_dims(aspect: str) -> tuple[int, int]:
    """Map common aspect ratios to FLUX-allowed (width, height)."""
    table = {
        "1:1": (1024, 1024),
        "16:9": (1344, 768),
        "9:16": (768, 1344),
        "4:3": (1152, 896),
        "3:4": (896, 1152),
        "3:2": (1216, 832),
        "2:3": (832, 1216),
    }
    return table.get(aspect, (1024, 1024))


class NvidiaClient:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("NVIDIA_API_KEY")
        if not self.api_key:
            raise RuntimeError("NVIDIA_API_KEY not set")
        self.llm_client = OpenAI(base_url=LLM_BASE_URL, api_key=self.api_key)

    def chat(
        self,
        messages: list[dict],
        model: str = DEFAULT_LLM,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> str:
        resp = self.llm_client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content or ""

    def chat_with_images(
        self,
        prompt: str,
        image_paths: list[str | Path],
        model: str = DEFAULT_VLM,
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ) -> str:
        content: list[dict] = [{"type": "text", "text": prompt}]
        for p in image_paths:
            b64 = _b64_image(p)
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{b64}"},
                }
            )
        return self.chat(
            [{"role": "user", "content": content}],
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    def upload_asset(self, image_path: str | Path, description: str = "ref") -> str:
        """Upload an image to NVCF assets. Returns asset_id."""
        r = requests.post(
            NVCF_ASSETS_URL,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "accept": "application/json",
                "Content-Type": "application/json",
            },
            json={"contentType": "image/png", "description": description},
            timeout=60,
        )
        r.raise_for_status()
        info = r.json()
        upload_url = info["uploadUrl"]
        asset_id = info["assetId"]

        put = requests.put(
            upload_url,
            data=Path(image_path).read_bytes(),
            headers={
                "Content-Type": "image/png",
                "x-amz-meta-nvcf-asset-description": description,
            },
            timeout=180,
        )
        put.raise_for_status()
        return asset_id

    def generate_image(
        self,
        prompt: str,
        model: str = DEFAULT_IMAGE_MODEL,
        seed: Optional[int] = None,
        aspect_ratio: str = "16:9",
        cfg_scale: float = 3.5,
        steps: int = 50,
        ref_image_path: Optional[str | Path] = None,
    ) -> bytes:
        """Generate an image. If ref_image_path given, uses kontext (img2img).
        Falls back to plain text2img on kontext failure.
        """
        url = f"{IMAGE_BASE_URL}/{model}"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        }

        if ref_image_path and model == DEFAULT_KONTEXT_MODEL:
            asset_id = self.upload_asset(ref_image_path)
            headers["NVCF-INPUT-ASSET-REFERENCES"] = asset_id
            payload: dict = {
                "prompt": prompt,
                "image": f"data:image/png;example_id,{asset_id}",
                "cfg_scale": cfg_scale,
                "aspect_ratio": "match_input_image",
            }
            if seed is not None:
                payload["seed"] = seed
        else:
            w, h = aspect_ratio_to_dims(aspect_ratio)
            payload = {
                "prompt": prompt,
                "width": w,
                "height": h,
                "cfg_scale": cfg_scale,
                "steps": steps,
            }
            if seed is not None:
                payload["seed"] = seed

        r = requests.post(url, headers=headers, json=payload, timeout=240)
        if r.status_code >= 400:
            raise RuntimeError(
                f"Image gen failed [{r.status_code}] model={model}: {r.text[:400]}"
            )
        data = r.json()
        artifacts = data.get("artifacts") or []
        if artifacts and artifacts[0].get("base64"):
            return base64.b64decode(artifacts[0]["base64"])
        b64 = data.get("image")
        if b64:
            return base64.b64decode(b64)
        raise RuntimeError(f"No image in response: {str(data)[:300]}")
