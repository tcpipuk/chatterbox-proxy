"""FastAPI adapter for Chatterbox TTS API, providing OpenAI-compatible endpoints.

This adapter lets any app using OpenAI's TTS API tap into Chatterbox's awesome voice cloning powers.
It's like a smart translator that converts OpenAI's simple requests into Chatterbox's more powerful
format, handling all the voice settings behind the scenes.

What makes this cool:
- Clone any voice from a single audio sample (yes, really! 🎯)
- Control how emotional the voice sounds
- Generate studio-quality audio
- Works with any OpenAI TTS client - no code changes needed
- Easy to add new voices or tweak existing ones

The adapter keeps track of all your voice presets and their settings, making it super simple to
experiment with different voices and emotional styles.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import httpx
import yaml
from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, Field

app = FastAPI()
logger = logging.getLogger("uvicorn.error")

CHATTERBOX_URL = f"http://{os.getenv('CHATTER_HOST', 'localhost')}:{os.getenv('CHATTER_PORT', '8000')}/speech"
MAX_INPUT_LENGTH = 500
VOICES_FILE = os.getenv("VOICES_FILE", "voices.yml")
DEFAULT_CONTENT_TYPE = "application/octet-stream"


class VoicePreset(BaseModel):
    """Voice preset configuration."""

    name: str
    audio_prompt: str | None = Field(default=None)
    exaggeration: float = Field(default=0.5, ge=0.0, le=1.0)
    cfg: float = Field(default=0.5, ge=0.0, le=1.0)
    temperature: float | None = Field(default=None, ge=0.0, le=1.0)


class TTSRequest(BaseModel):
    """OpenAI-compatible TTS request model.

    This is what your app sends to the proxy. The model field is ignored (we always use Chatterbox),
    but we keep it to maintain compatibility with OpenAI's API.
    """

    model: str | None = "tts-1"
    voice: str
    input: str


def load_voice_presets() -> dict[str, VoicePreset]:
    """Load voice presets from YAML file.

    Returns:
        Dictionary of voice presets with their settings.

    Raises:
        FileNotFoundError: If the voices file doesn't exist.
        yaml.YAMLError: If the YAML file is invalid.
    """
    try:
        with Path(VOICES_FILE).open(encoding="utf-8") as f:
            raw_presets = yaml.safe_load(f)

        presets: dict[str, VoicePreset] = {}
        for voice_id, preset in raw_presets.items():
            # Ensure audio_prompt is relative to reference/ directory if provided
            if preset.get("audio_prompt") and not preset["audio_prompt"].startswith("reference/"):
                preset["audio_prompt"] = f"reference/{preset['audio_prompt']}"

            # Validate and convert to VoicePreset model
            presets[voice_id] = VoicePreset(**preset)
    except FileNotFoundError:
        logger.exception("Voices file not found: %s", VOICES_FILE)
        raise
    except yaml.YAMLError:
        logger.exception("Invalid YAML in voices file")
        raise
    else:
        return presets


# Load voice presets at startup
try:
    VOICE_PRESETS = load_voice_presets()
except (FileNotFoundError, yaml.YAMLError, ValueError):
    logger.exception("Failed to load voice presets")
    VOICE_PRESETS = {}  # Fallback to empty dict if loading fails


@app.get("/health")
async def health_check() -> dict[str, str]:
    """Health check endpoint.

    Returns:
        Status message indicating the service is healthy.
    """
    return {"status": "healthy"}


@app.get("/v1/audio/voices")
async def list_voices() -> list[dict[str, str]]:
    """List all available voice presets.

    Returns:
        A list of voice presets, each with an ID and display name. These are the voices
        your app can use when making TTS requests.
    """
    return [{"id": k, "name": v.name} for k, v in VOICE_PRESETS.items()]


@app.get("/v1/audio/models")
async def list_models() -> list[dict[str, str | int]]:
    """List available TTS models.

    Returns:
        A list containing the Chatterbox model info to maintain compatibility with OpenAI's API.
        We always use Chatterbox under the hood, but this endpoint lets your app check
        what's available.
    """
    return [{"id": "chatterbox", "created": 1713956629}]


@app.post("/v1/audio/speech")
async def openai_tts(req: TTSRequest) -> Response:
    """Generate speech from text using Chatterbox TTS.

    This is where the magic happens! The function:
    1. Checks if your request is valid
    2. Finds the right voice preset
    3. Converts the request to Chatterbox format
    4. Sends it to Chatterbox
    5. Returns the generated audio

    Args:
        req: Your TTS request with the text to speak and which voice to use.

    Returns:
        The generated audio file, ready to play.

    Raises:
        HTTPException: If the text is too long or the voice isn't found.
    """
    logger.info(
        "Received TTS request: model=%s, voice=%s, input_length=%d", req.model, req.voice, len(req.input)
    )

    if len(req.input) > MAX_INPUT_LENGTH:
        logger.warning("Input text too long: %d > %d", len(req.input), MAX_INPUT_LENGTH)
        raise HTTPException(
            status_code=400,
            detail=f"Input text exceeds {MAX_INPUT_LENGTH} characters; please shorten.",
        )

    if req.voice not in VOICE_PRESETS:
        logger.warning("Unknown voice preset: %s (available: %s)", req.voice, ", ".join(VOICE_PRESETS.keys()))
        raise HTTPException(
            status_code=400,
            detail=f"Unknown voice preset: {req.voice}",
        )

    preset = VOICE_PRESETS[req.voice]
    logger.info(
        "Using voice preset: %s (audio_prompt=%s, exaggeration=%.1f, cfg=%.1f, temperature=%s)",
        preset.name,
        preset.audio_prompt,
        preset.exaggeration,
        preset.cfg,
        preset.temperature,
    )

    # Build payload with required fields
    payload = {
        "text": req.input,
        "exaggeration": preset.exaggeration,
        "cfg": preset.cfg,
    }

    # Only include audio_prompt or temperature, not both
    if preset.audio_prompt:
        payload["audio_prompt"] = preset.audio_prompt
    elif preset.temperature is not None:
        payload["temperature"] = preset.temperature

    logger.info("Sending request to Chatterbox: %s", payload)

    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            chatterbox_response = await client.post(CHATTERBOX_URL, json=payload)
            chatterbox_response.raise_for_status()
        except httpx.HTTPStatusError as e:
            status = e.response.status_code
            detail = f"Chatterbox error: {e.response.text}"
            logger.exception("Chatterbox request failed: %s (status=%d)", detail, status)
            raise HTTPException(status_code=status, detail=detail) from e
        except httpx.RequestError as e:
            logger.exception("Failed to connect to Chatterbox")
            raise HTTPException(
                status_code=503,
                detail="Chatterbox service unavailable",
            ) from e
        except Exception as e:
            logger.exception("Unexpected error during TTS request")
            raise HTTPException(
                status_code=500,
                detail="Internal server error",
            ) from e

    content_type = chatterbox_response.headers.get("content-type", DEFAULT_CONTENT_TYPE)
    logger.info("Successfully generated audio (content-type: %s)", content_type)
    return Response(content=chatterbox_response.content, media_type=content_type)
