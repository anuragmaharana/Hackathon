#!/usr/bin/env python3
"""Run the AI Image Studio pipeline on the provided input image.

Example:
    python run.py --input image.jpg --output output.jpg \
        --edit-api-url "https://..." --edit-api-key "..." \
        --upscale-api-url "https://..." --upscale-api-key "..."

The upscale step defaults to running Real-ESRGAN-x4plus locally, in-process,
via qai_hub_models (`pip install "qai-hub-models[real-esrgan-x4plus]"`).
Pass --no-local-upscaler to skip that and use the remote --upscale-api-url
(or a plain resize if none is set) instead.

If you prefer, set the same values as environment variables instead:
    EDIT_API_URL, EDIT_API_KEY, FALLBACK_EDIT_API_URL, FALLBACK_EDIT_API_KEY,
    UPSCALE_API_URL, UPSCALE_API_KEY
"""

from __future__ import annotations

import argparse
import getpass
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    from apikeys import gemini_key
except ImportError:  # pragma: no cover - optional local credentials module
    gemini_key = ""

from ai_Image_Studio import APIConfig, process_image


def _get_value(cli_value: str | None, env_name: str, default: str = "") -> str:
    if cli_value is not None:
        return cli_value
    return os.getenv(env_name, default)


def _load_gemini_key() -> str:
    if gemini_key.strip():
        return gemini_key.strip()

    key_file = Path(__file__).with_name("Api keys.txt")
    if not key_file.exists():
        return ""
    for line in key_file.read_text(encoding="utf-8").splitlines():
        value = line.strip()
        if value and not value.startswith("#"):
            if value.lower().startswith("gemini key"):
                value = value[len("gemini key"):].strip(" :\t")
            return value
    return ""


def _prompt_value(label: str, current: str = "", secret: bool = False) -> str:
    """Return an existing value or ask for it interactively."""
    if current:
        return current
    prompt = f"{label}: "
    return getpass.getpass(prompt) if secret else input(prompt).strip()


def _collect_api_settings(args: argparse.Namespace, interactive: bool) -> dict[str, str]:
    """Collect missing API settings for an interactive CLI run."""
    default_edit_url = (
        "https://generativelanguage.googleapis.com/v1beta/"
        "models/gemini-2.5-flash-image-preview:generateContent"
    )
    edit_url = _get_value(args.edit_api_url, "EDIT_API_URL", default_edit_url)
    edit_key = _get_value(args.edit_api_key, "EDIT_API_KEY", _load_gemini_key())
    fallback_url = _get_value(args.fallback_edit_api_url, "FALLBACK_EDIT_API_URL")
    fallback_key = _get_value(args.fallback_edit_api_key, "FALLBACK_EDIT_API_KEY")
    upscale_url = _get_value(args.upscale_api_url, "UPSCALE_API_URL")
    upscale_key = _get_value(args.upscale_api_key, "UPSCALE_API_KEY")

    if not interactive or not sys.stdin.isatty():
        if not edit_key:
            raise SystemExit(
                "Missing EDIT_API_KEY. Set it in the environment, Api keys.txt, "
                "or pass --edit-api-key."
            )
        return {
            "edit_url": edit_url,
            "edit_key": edit_key,
            "fallback_url": fallback_url,
            "fallback_key": fallback_key,
            "upscale_url": upscale_url,
            "upscale_key": upscale_key,
        }

    print("\nAPI setup (press Enter to keep a shown default or skip an optional service).")
    print("Required: one image-edit API URL and key (Gemini is the default).")
    if args.no_local_upscaler:
        print("Optional: fallback edit API and remote upscaling API; local resize is used when skipped.\n")
    else:
        print("Upscaling will run locally via qai_hub_models (Real-ESRGAN-x4plus); "
              "no remote upscale API is needed unless you passed --no-local-upscaler.\n")
    edit_url = _prompt_value("Primary edit API URL", edit_url)
    edit_key = _prompt_value("Primary edit API key", edit_key, secret=True)
    if not edit_key:
        raise SystemExit("A primary edit API key is required.")
    fallback_url = _prompt_value("Fallback edit API URL (optional)", fallback_url)
    if fallback_url:
        fallback_key = _prompt_value("Fallback edit API key (optional)", fallback_key, secret=True)
    if args.no_local_upscaler:
        upscale_url = _prompt_value("Upscale API URL (optional)", upscale_url)
        if upscale_url:
            upscale_key = _prompt_value("Upscale API key (optional)", upscale_key, secret=True)

    return {
        "edit_url": edit_url,
        "edit_key": edit_key,
        "fallback_url": fallback_url,
        "fallback_key": fallback_key,
        "upscale_url": upscale_url,
        "upscale_key": upscale_key,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the AI Image Studio pipeline.")
    parser.add_argument("--input", default="image.jpg", help="Input image path")
    parser.add_argument("--output", default="output.jpg", help="Output image path")
    parser.add_argument("--edit-api-url", default=None, help="Primary edit API URL")
    parser.add_argument("--edit-api-key", default=None, help="Primary edit API key")
    parser.add_argument("--fallback-edit-api-url", default=None, help="Fallback edit API URL")
    parser.add_argument("--fallback-edit-api-key", default=None, help="Fallback edit API key")
    parser.add_argument("--edit-model", default="gemini-image-edit", help="Primary edit model")
    parser.add_argument("--fallback-edit-model", default="flux-kontext", help="Fallback edit model")
    parser.add_argument(
        "--no-local-upscaler",
        action="store_true",
        help="Skip the local qai_hub_models Real-ESRGAN upscaler and use the "
             "remote --upscale-api-url (or a plain resize) instead",
    )
    parser.add_argument(
        "--local-upscale-model",
        default="real_esrgan_x4plus",
        help="qai_hub_models module name to use for local upscaling "
             "(e.g. real_esrgan_x4plus, real_esrgan_general_x4v3)",
    )
    parser.add_argument("--upscale-api-url", default=None, help="Upscale API URL")
    parser.add_argument("--upscale-api-key", default=None, help="Upscale API key")
    parser.add_argument("--upscale-model", default="real-esrgan-x4", help="Upscale model")
    parser.add_argument(
        "--target-size",
        default=None,
        help="Optional final crop size like 2000x2000 or 2000,2000",
    )
    return parser.parse_args()


def parse_target_size(value: str | None) -> tuple[int, int] | None:
    if value is None:
        return None
    if "x" in value.lower():
        parts = value.lower().split("x", 1)
    else:
        parts = value.split(",", 1)
    if len(parts) != 2:
        raise ValueError("--target-size must be like 2000x2000 or 2000,2000")
    width, height = (int(p.strip()) for p in parts)
    return (width, height)


def run_pipeline(args: argparse.Namespace | None = None) -> str:
    """Invoke the pipeline programmatically (or via CLI when args is None).

    Returns the output path on success.
    """
    interactive = args is None
    if args is None:
        args = parse_args()

    try:
        target_size = parse_target_size(args.target_size)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    api_settings = _collect_api_settings(args, interactive=interactive)
    cfg = APIConfig(
        edit_api_url=api_settings["edit_url"],
        edit_api_key=api_settings["edit_key"],
        fallback_edit_api_url=api_settings["fallback_url"],
        fallback_edit_api_key=api_settings["fallback_key"],
        edit_model=args.edit_model,
        fallback_edit_model=args.fallback_edit_model,
        use_local_qai_upscaler=not args.no_local_upscaler,
        local_qai_model_name=args.local_upscale_model,
        upscale_api_url=api_settings["upscale_url"],
        upscale_api_key=api_settings["upscale_key"],
        upscale_model=args.upscale_model,
    )

    result_path = process_image(args.input, args.output, cfg, target_size=target_size)
    print(f"Pipeline completed successfully: {result_path}")
    return result_path


if __name__ == "__main__":
    run_pipeline()