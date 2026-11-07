"""Serves Cloudflare's Clef-flash model as a local System One HTTP API.

This is a proof of concept. The weights are loaded from the local Hugging Face
cache, so nothing is downloaded. See README.md for an example request.

Typical usage:

    source .venv/bin/activate
    uvicorn app:app --port 8788
"""

import base64
from collections.abc import AsyncIterator
import contextlib
import importlib
import io
import pathlib
import sys
import threading
from typing import Any

import fastapi
from PIL import Image
from PIL import ImageOps
import torch

_HF_HUB = pathlib.Path.home() / ".cache/huggingface/hub"
_REPO = _HF_HUB / "models--Cloudflare--clef-flash"
MODEL_PATH = _REPO / "snapshots" / (_REPO / "refs/main").read_text().strip()
# Use torch.float16 if MPS complains about bfloat16.
DTYPE = torch.bfloat16
# Images are downscaled to about 0.5 MP: the measured sweet spot between speed
# and accuracy (see README.md). The hosted service uses about 1 MP.
MAX_PIXELS = 512 * 1024
MAX_IMAGES = 4


def _pick_device() -> str:
    """Returns the best available torch device."""
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


DEVICE = _pick_device()


def _decode_image(item: str | dict[str, Any]) -> Image.Image:
    """Decodes one image in the Workers AI format and downscales it.

    Args:
        item: A data URL ("data:image/png;base64,...") or a dict with a
            "base64" key, such as {"content_type": "image/png", "base64": ...}.

    Returns:
        An RGB image of at most about MAX_PIXELS pixels.

    Raises:
        ValueError: The item is in neither format or is not valid base64.
        OSError: The bytes are not an image that PIL can read.
    """
    if isinstance(item, str) and item.startswith("data:") and "," in item:
        raw = base64.b64decode(item.split(",", 1)[1])
    elif isinstance(item, dict) and "base64" in item:
        raw = base64.b64decode(item["base64"])
    else:
        raise ValueError(
            "each image must be a data URL or {content_type, base64}"
        )
    img = ImageOps.exif_transpose(Image.open(io.BytesIO(raw))).convert("RGB")
    area = img.width * img.height
    if area <= MAX_PIXELS:
        return img
    scale = (MAX_PIXELS / area) ** 0.5
    size = (max(32, int(img.width * scale)), max(32, int(img.height * scale)))
    return img.resize(size, Image.Resampling.BICUBIC)


@contextlib.asynccontextmanager
async def _lifespan(application: fastapi.FastAPI) -> AsyncIterator[None]:
    """Loads the model once at startup and keeps it on application.state."""
    # Cloudflare's model code ships with the weights, so import it from there.
    sys.path.insert(0, str(MODEL_PATH))
    clef = importlib.import_module("joint_schema_model")
    model, processor = clef.load_release_model(
        str(MODEL_PATH), device=DEVICE, dtype=DTYPE
    )
    model.eval()
    application.state.clef = clef
    application.state.model = model
    application.state.processor = processor
    # The GPU is the bottleneck, so run one request at a time.
    application.state.lock = threading.Lock()
    yield


app = fastapi.FastAPI(title="Clef System One", lifespan=_lifespan)


@app.get("/health")
def health() -> dict[str, Any]:
    """Reports that the server is up, and which model and device it uses."""
    return {"ok": True, "model": MODEL_PATH.name, "device": DEVICE}


# A plain `def` endpoint runs in FastAPI's threadpool, so inference doesn't
# block the event loop.
@app.post("/v1/systemone")
def systemone(
    body: dict[str, Any], request: fastapi.Request
) -> dict[str, Any]:
    """Answers a System One request in the Cloudflare Workers AI format.

    The body has "state", "questions" and up to 4 "images". See README.md.
    """
    images = body.get("images") or []
    if len(images) > MAX_IMAGES:
        raise fastapi.HTTPException(400, f"at most {MAX_IMAGES} images")
    state = request.app.state
    try:
        decoded = [_decode_image(item) for item in images]
        clef_request = {**body, "model": MODEL_PATH.name, "images": decoded}
        with state.lock:
            result = state.clef.systemone(
                state.model, state.processor, clef_request
            )
    except (ValueError, OSError) as e:  # Bad images or questions.
        raise fastapi.HTTPException(400, str(e)) from e
    return {"success": True, "errors": [], "messages": [], "result": result}
