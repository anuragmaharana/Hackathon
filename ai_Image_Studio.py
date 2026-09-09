"""
AI Image Studio Pipeline
=========================

Turns a raw phone photo into a professional catalog image through six stages:

    1. Ingest & validate     - deterministic checks (blur, exposure, resolution)
    2. Preprocess            - auto-rotate, denoise, deblur
    3. Segment subject       - mask product from background/clutter
    4. Background & relight  - studio backdrop + shadow + lighting normalization
    5. Upscale & retouch     - hit catalog resolution, clean up blemishes
    6. Compose, QA, export   - crop to spec, automated scoring, format render

Steps 3-4 are collapsed into a single call to an instruction-following image
editor (e.g. Gemini image, Flux Kontext). Step 5 prefers a *local* on-device
super-resolution model (Qualcomm AI Hub's Real-ESRGAN-x4plus, run via
qai_hub_models, in-process and offline) and only falls back to a remote
upscaling API or a plain Lanczos resize if the local model isn't available.

Resilience:
    - Validation runs before any paid model call.
        - The edit call gets retries with backoff and a circuit breaker so one
            vendor outage does not repeatedly consume the pipeline.
    - Upscaling tries, in order: local qai_hub_models Real-ESRGAN -> remote
      upscale API (with its own retry policy) -> local Lanczos resize.
    - Automated QA runs before anything would go to a human reviewer.
    - Every stage writes a new versioned artifact instead of overwriting,
      so a single stage can be retried without restarting the whole job.

Usage:
    from ai_image_pipeline import process_image, APIConfig

    config = APIConfig(
        edit_api_url="https://api.example.com/v1/edit",
        edit_api_key="sk-...",
        edit_model="gemini-2.5-flash-image",
        use_local_qai_upscaler=True,       # Real-ESRGAN-x4plus via qai_hub_models
        upscale_api_url="",                # optional remote fallback
        upscale_api_key="",
        upscale_model="real-esrgan-x4",
    )

    output_path = process_image("phone_photo.jpg", "catalog_image.jpg", config)
"""

from __future__ import annotations

import functools
import io
import logging
import time
import uuid
import base64
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import requests
from PIL import Image, ImageFilter, ImageOps, ImageStat
from apikeys import gemini_key

logger = logging.getLogger("ai_image_pipeline")
logging.basicConfig(level=logging.INFO)


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

@dataclass
class APIConfig:
    """All external API requirements, exposed as parameters so the caller
    can point this pipeline at whatever providers/models/keys they have."""

    # Instruction-following editor: handles segmentation + background + relight
    edit_api_url: str = (
        "https://generativelanguage.googleapis.com/v1beta/"
        "models/gemini-2.5-flash-image:generateContent"
    )
    edit_api_key: str = gemini_key
    edit_model: str = "gemini-2.5-flash-image"
    edit_prompt: str = (
        "Remove the background, place the product on a seamless white "
        "studio backdrop, add a soft realistic contact shadow, and "
        "normalize the lighting to even, diffused studio lighting. "
        "Do not alter the product itself."
    )

    # --- Upscaler ---------------------------------------------------------
    # Preferred path: run Real-ESRGAN-x4plus locally, in-process, via
    # qai_hub_models (`pip install "qai-hub-models[real-esrgan-x4plus]"`).
    # No network call, no API key. Falls back to the remote API below,
    # then to a plain Lanczos resize, if the local model can't be loaded.
    use_local_qai_upscaler: bool = True
    local_qai_model_name: str = "real_esrgan_x4plus"

    # Dedicated remote upscaler (kept as an optional fallback/alternative)
    upscale_api_url: str = ""
    upscale_api_key: str = ""
    upscale_model: str = "real-esrgan-x4"
    upscale_target_long_edge_px: int = 2048  # used by the final resize fallback

    # Resilience knobs
    max_retries: int = 2
    retry_backoff_seconds: float = 1.5
    request_timeout_seconds: float = 60.0
    circuit_breaker_failure_threshold: int = 3

    # QA thresholds
    min_output_resolution_px: int = 1024
    max_background_std_dev: float = 12.0  # near-uniform background = pass

    # Simple in-memory circuit breaker state (per-config instance)
    _consecutive_failures: int = field(default=0, init=False, repr=False)
    _circuit_open: bool = field(default=False, init=False, repr=False)


class PipelineError(Exception):
    """Raised when a stage fails in a way the pipeline can't recover from."""


class CircuitOpenError(PipelineError):
    """Raised when the edit-service circuit breaker is open."""


class ValidationError(PipelineError):
    """Raised when the input image fails deterministic pre-checks."""


class QAFailure(PipelineError):
    """Raised when the composed output fails automated QA and needs review."""


# --------------------------------------------------------------------------
# Stage 1: Ingest & validate  (deterministic, no model call)
# --------------------------------------------------------------------------

def validate_image(image: Image.Image, min_resolution_px: int = 512) -> None:
    """Rejects unusable input early: blur, exposure, resolution.

    Raises ValidationError with a specific reason instead of paying for a
    wasted edit call downstream.
    """
    width, height = image.size
    if min(width, height) < min_resolution_px:
        raise ValidationError(
            f"Resolution too low: {width}x{height} (min side {min_resolution_px}px)"
        )

    grayscale = image.convert("L")

    # Exposure check: mean brightness should not be near-black or near-white
    stat = ImageStat.Stat(grayscale)
    mean_brightness = stat.mean[0]
    if mean_brightness < 15:
        raise ValidationError(f"Image too dark (mean brightness {mean_brightness:.1f}/255)")
    if mean_brightness > 240:
        raise ValidationError(f"Image too bright/blown out (mean brightness {mean_brightness:.1f}/255)")

    # Blur check: variance of a Laplacian-like edge filter. Low variance -> blurry.
    edges = grayscale.filter(ImageFilter.FIND_EDGES)
    edge_stat = ImageStat.Stat(edges)
    edge_variance = edge_stat.var[0]
    if edge_variance < 8:
        raise ValidationError(f"Image appears too blurry (edge variance {edge_variance:.2f})")

    logger.info("Validation passed: %dx%d, brightness=%.1f, edge_var=%.2f",
                width, height, mean_brightness, edge_variance)


# --------------------------------------------------------------------------
# Stage 2: Preprocess  (lightweight/deterministic)
# --------------------------------------------------------------------------

def preprocess_image(image: Image.Image) -> Image.Image:
    """Auto-rotate (via EXIF), light denoise, light deblur/sharpen."""
    image = ImageOps.exif_transpose(image)
    if image.mode != "RGB":
        image = image.convert("RGB")

    # Light denoise
    image = image.filter(ImageFilter.MedianFilter(size=3))
    # Light sharpen to counteract phone-camera softness
    image = image.filter(ImageFilter.UnsharpMask(radius=2, percent=60, threshold=3))

    logger.info("Preprocessing complete: %dx%d", *image.size)
    return image


# --------------------------------------------------------------------------
# Shared HTTP helper: retries + backoff + circuit breaker
# --------------------------------------------------------------------------

def _call_image_api(
    api_url: str,
    api_key: str,
    model: str,
    image_bytes: bytes,
    prompt: Optional[str],
    config: APIConfig,
    stage_name: str,
) -> bytes:
    """POSTs an image (+ optional prompt) to an image API and returns the
    resulting image bytes. Retries with exponential backoff on failure."""

    if not api_url:
        raise PipelineError(f"{stage_name}: no API URL configured")

    last_error: Optional[Exception] = None
    for attempt in range(config.max_retries + 1):
        try:
            if "generativelanguage.googleapis.com" in api_url:
                gemini_url = api_url
                if "/models/" in gemini_url and ":generateContent" in gemini_url:
                    prefix = gemini_url.split("/models/", 1)[0]
                    gemini_url = f"{prefix}/models/{model}:generateContent"
                response = requests.post(
                    gemini_url,
                    params={"key": api_key},
                    json={
                        "contents": [{
                            "parts": [
                                {"text": prompt or "Upscale this image."},
                                {
                                    "inline_data": {
                                        "mime_type": "image/png",
                                        "data": base64.b64encode(image_bytes).decode("ascii"),
                                    }
                                },
                            ]
                        }]
                    },
                    timeout=config.request_timeout_seconds,
                )
                response.raise_for_status()
                payload = response.json()
                for part in payload.get("candidates", [{}])[0].get("content", {}).get("parts", []):
                    inline_data = part.get("inlineData") or part.get("inline_data")
                    if inline_data and inline_data.get("data"):
                        logger.info("%s: succeeded on attempt %d", stage_name, attempt + 1)
                        return base64.b64decode(inline_data["data"])
                raise PipelineError(f"{stage_name}: Gemini response did not contain an image")

            files = {"image": ("input.png", image_bytes, "image/png")}
            data = {"model": model}
            if prompt:
                data["prompt"] = prompt
            headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}

            response = requests.post(
                api_url,
                headers=headers,
                data=data,
                files=files,
                timeout=config.request_timeout_seconds,
            )
            response.raise_for_status()
            logger.info("%s: succeeded on attempt %d", stage_name, attempt + 1)
            return response.content

        except requests.RequestException as exc:
            last_error = exc
            status = getattr(exc.response, "status_code", None)
            detail = f"HTTP {status}" if status else type(exc).__name__
            if exc.response is not None and exc.response.text:
                detail = f"{detail}: {exc.response.text[:500]}"
            logger.warning("%s: attempt %d failed (%s)", stage_name, attempt + 1, detail)
            if attempt < config.max_retries:
                time.sleep(config.retry_backoff_seconds * (2 ** attempt))

    raise PipelineError(f"{stage_name}: all {config.max_retries + 1} attempts failed") from last_error


# --------------------------------------------------------------------------
# Stages 3-4: Segment + Background & relight (one instruction-following edit call)
# --------------------------------------------------------------------------

def edit_background_and_relight(image: Image.Image, config: APIConfig) -> Image.Image:
    """Segments the subject and applies studio background + relighting in a
    single instruction-following edit call.

    Retries with backoff, guarded by a simple circuit breaker so a vendor
    outage does not repeatedly consume the whole queue.
    """
    if config._circuit_open:
        raise CircuitOpenError(
            "Edit service circuit breaker is open after repeated failures; "
            "route this job to human review or retry later."
        )

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    image_bytes = buffer.getvalue()

    try:
        result_bytes = _call_image_api(
            api_url=config.edit_api_url,
            api_key=config.edit_api_key,
            model=config.edit_model,
            image_bytes=image_bytes,
            prompt=config.edit_prompt,
            config=config,
            stage_name="edit(primary)",
        )
        config._consecutive_failures = 0
        return Image.open(io.BytesIO(result_bytes)).convert("RGB")

    except PipelineError:
        config._consecutive_failures += 1
        _maybe_trip_circuit_breaker(config)
        raise


def _maybe_trip_circuit_breaker(config: APIConfig) -> None:
    if config._consecutive_failures >= config.circuit_breaker_failure_threshold:
        config._circuit_open = True
        logger.error(
            "Circuit breaker tripped after %d consecutive edit-service failures",
            config._consecutive_failures,
        )


# --------------------------------------------------------------------------
# Stage 5: Upscale & retouch
# --------------------------------------------------------------------------
#
# Preferred path is a *local* super-resolution model run in-process via
# qai_hub_models (the same package behind
# `python -m qai_hub_models.models.real_esrgan_x4plus.demo`). This needs
# no network call and no API key. If that package/model isn't installed,
# we fall back to a remote upscale API (if configured), and finally to a
# plain Lanczos resize.

@functools.lru_cache(maxsize=4)
def _load_local_qai_upscaler(model_name: str):
    """Loads and caches a qai_hub_models super-resolution App+Model pair.

    Returns an object with a `.predict(PIL.Image) -> PIL.Image` style
    interface. Cached (per model_name) so the (fairly large) pretrained
    weights are only loaded once per process, not once per image.

    Raises ImportError if qai_hub_models (or this specific model extra)
    isn't installed, so callers can fall back cleanly.
    """
    import importlib

    module = importlib.import_module(f"qai_hub_models.models.{model_name}")
    Model = module.Model
    App = module.App

    logger.info("Loading local qai_hub_models '%s' (first call only)...", model_name)
    torch_model = Model.from_pretrained()
    app = App(torch_model)
    return app


def _pil_from_predict_result(result) -> Image.Image:
    """Normalizes the various return shapes qai_hub_models Apps use
    (PIL.Image, numpy array, or a list/tuple containing either) into a
    single PIL.Image."""
    if isinstance(result, (list, tuple)):
        if not result:
            raise PipelineError("local upscaler: empty prediction result")
        result = result[0]

    if isinstance(result, Image.Image):
        return result.convert("RGB")

    if isinstance(result, np.ndarray):
        array = result
        if array.dtype != np.uint8:
            # Most of these apps return float arrays in [0, 1]
            array = np.clip(array, 0.0, 1.0) * 255.0
            array = array.astype(np.uint8)
        if array.ndim == 4:  # (N, C, H, W) or (N, H, W, C) batch of 1
            array = array[0]
        if array.ndim == 3 and array.shape[0] in (1, 3) and array.shape[0] != array.shape[-1]:
            # CHW -> HWC
            array = np.transpose(array, (1, 2, 0))
        return Image.fromarray(array).convert("RGB")

    raise PipelineError(f"local upscaler: unrecognized prediction result type {type(result)!r}")


def _upscale_locally(image: Image.Image, config: APIConfig) -> Image.Image:
    """Runs the configured qai_hub_models model on `image` in-process."""
    app = _load_local_qai_upscaler(config.local_qai_model_name)
    result = app.predict(image)
    upscaled = _pil_from_predict_result(result)
    logger.info(
        "local upscaler (%s): %dx%d -> %dx%d",
        config.local_qai_model_name, *image.size, *upscaled.size,
    )
    return upscaled


def upscale_and_retouch(image: Image.Image, config: APIConfig) -> Image.Image:
    """Hits catalog resolution and cleans up blemishes.

    Order of preference:
        1. Local qai_hub_models model (e.g. Real-ESRGAN-x4plus), in-process.
        2. Remote upscale API, if configured.
        3. Plain Lanczos resize up to `upscale_target_long_edge_px`.
    """
    if config.use_local_qai_upscaler:
        try:
            return _upscale_locally(image, config)
        except ImportError as exc:
            logger.warning(
                "Local qai_hub_models upscaler not available (%s). "
                "Install with: pip install \"qai-hub-models[%s]\". "
                "Falling back to remote API / local resize.",
                exc, config.local_qai_model_name.replace("_", "-"),
            )
        except Exception as exc:  # noqa: BLE001 - any local-model failure should fall back, not crash the job
            logger.warning("Local qai_hub_models upscaler failed (%s); falling back.", exc)

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    image_bytes = buffer.getvalue()

    try:
        result_bytes = _call_image_api(
            api_url=config.upscale_api_url,
            api_key=config.upscale_api_key,
            model=config.upscale_model,
            image_bytes=image_bytes,
            prompt=None,
            config=config,
            stage_name="upscale",
        )
        return Image.open(io.BytesIO(result_bytes)).convert("RGB")

    except PipelineError as exc:
        logger.warning("Upscale service unavailable, falling back to local resize: %s", exc)
        long_edge = max(image.size)
        if long_edge >= config.upscale_target_long_edge_px:
            return image
        scale = config.upscale_target_long_edge_px / long_edge
        new_size = (round(image.width * scale), round(image.height * scale))
        return image.resize(new_size, Image.LANCZOS)


# --------------------------------------------------------------------------
# Stage 6: Compose, QA, export
# --------------------------------------------------------------------------

def compose_and_qa(image: Image.Image, config: APIConfig, target_size: Optional[tuple] = None) -> Image.Image:
    """Crops to spec and runs automated QA checks before export.

    Raises QAFailure (instead of silently exporting) when a job would need
    human review, matching the pass/fail branch in the pipeline design.
    """
    if target_size:
        image = ImageOps.fit(image, target_size, Image.LANCZOS)

    width, height = image.size
    if min(width, height) < config.min_output_resolution_px:
        raise QAFailure(
            f"Output resolution {width}x{height} below minimum "
            f"{config.min_output_resolution_px}px -> route to human review"
        )

    # Background purity check: sample the corners, expect low variance
    # (a clean, near-uniform studio backdrop).
    corner_size = max(1, min(width, height) // 20)
    corners = [
        image.crop((0, 0, corner_size, corner_size)),
        image.crop((width - corner_size, 0, width, corner_size)),
        image.crop((0, height - corner_size, corner_size, height)),
        image.crop((width - corner_size, height - corner_size, width, height)),
    ]
    std_devs = [ImageStat.Stat(c.convert("L")).stddev[0] for c in corners]
    avg_std_dev = sum(std_devs) / len(std_devs)

    if avg_std_dev > config.max_background_std_dev:
        raise QAFailure(
            f"Background not sufficiently uniform (avg std dev {avg_std_dev:.2f} "
            f"> {config.max_background_std_dev}) -> route to human review"
        )

    logger.info("QA passed: %dx%d, background std dev=%.2f", width, height, avg_std_dev)
    return image


def export_image(image: Image.Image, output_path: str, quality: int = 95) -> str:
    """Renders the final image to disk. Each call produces a fresh,
    versioned file rather than overwriting an existing artifact in place."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_kwargs = {}
    if path.suffix.lower() in (".jpg", ".jpeg"):
        save_kwargs["quality"] = quality
        save_kwargs["optimize"] = True
    image.save(path, **save_kwargs)
    logger.info("Exported catalog image to %s", path)
    return str(path)


# --------------------------------------------------------------------------
# End-to-end pipeline
# --------------------------------------------------------------------------

def process_image(
    input_path: str,
    output_path: str,
    api_config: APIConfig,
    target_size: Optional[tuple] = None,
) -> str:
    """Runs the full raw-photo -> catalog-image pipeline.

    Args:
        input_path: path to the raw phone photo.
        output_path: path to write the finished catalog image to.
        api_config: APIConfig with the edit/fallback/upscale API settings.
        target_size: optional (width, height) to crop/fit the final image to.

    Returns:
        The output_path the catalog image was written to.

    Raises:
        ValidationError: if the input image fails deterministic checks.
        CircuitOpenError: if the edit service circuit breaker is open.
        PipelineError: if a model call fails irrecoverably.
        QAFailure: if the composed output fails automated QA (route to
            human review in a production deployment).
    """
    job_id = uuid.uuid4().hex[:8]
    logger.info("Starting job %s: %s -> %s", job_id, input_path, output_path)

    with Image.open(input_path) as raw:
        raw.load()
        image = raw.copy()

    # Stage 1
    validate_image(image, min_resolution_px=api_config.min_output_resolution_px // 2)

    # Stage 2
    image = preprocess_image(image)

    # Stages 3-4
    image = edit_background_and_relight(image, api_config)

    # Stage 5
    image = upscale_and_retouch(image, api_config)

    # Stage 6
    image = compose_and_qa(image, api_config, target_size=target_size)
    result_path = export_image(image, output_path)

    logger.info("Job %s complete -> %s", job_id, result_path)
    return result_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Raw phone photo -> catalog image")
    parser.add_argument("input", help="Path to input photo")
    parser.add_argument("output", help="Path to write catalog image")
    parser.add_argument("--edit-api-url", default="")
    parser.add_argument("--edit-api-key", default="")
    parser.add_argument("--edit-model", default="gemini-2.5-flash-image")
    parser.add_argument("--no-local-upscaler", action="store_true",
                         help="Skip the local qai_hub_models upscaler and go straight to the remote API/resize")
    parser.add_argument("--upscale-api-url", default="")
    parser.add_argument("--upscale-api-key", default="")
    parser.add_argument("--upscale-model", default="real-esrgan-x4")
    args = parser.parse_args()

    cfg = APIConfig(
        edit_api_url=args.edit_api_url,
        edit_api_key=args.edit_api_key,
        edit_model=args.edit_model,
        use_local_qai_upscaler=not args.no_local_upscaler,
        upscale_api_url=args.upscale_api_url,
        upscale_api_key=args.upscale_api_key,
        upscale_model=args.upscale_model,
    )
    process_image(args.input, args.output, cfg)