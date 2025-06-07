"""FastAPI adapter for Chatterbox TTS API, providing OpenAI-compatible endpoints.

This adapter serves as a bridge between OpenAI's TTS API format and Chatterbox's more powerful
voice cloning capabilities. It allows any application using OpenAI's TTS API to seamlessly
switch to Chatterbox's voice cloning features without code changes.

Key features:
- Zero-shot voice cloning from a single audio sample
- Emotional control over generated speech
- Studio-quality audio output
- Drop-in replacement for OpenAI's TTS API
- Configurable voice presets with custom settings

The adapter maintains a collection of voice presets, each with its own configuration for
voice cloning, emotional intensity, and generation quality. This makes it easy to experiment
with different voices and speaking styles.
"""

from __future__ import annotations

from io import BytesIO
from logging import getLogger
from os import getenv as os_getenv
from pathlib import Path
from re import finditer as re_finditer
from time import perf_counter
from typing import TYPE_CHECKING

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from httpx import AsyncClient as HTTPXAsyncClient
from pydantic import BaseModel, Field
from pydub import AudioSegment
from pydub.exceptions import CouldntDecodeError
from yaml import YAMLError, safe_load as yaml_safe_load

from audio_filters import FILTER_CHAINS

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

app = FastAPI()
logger = getLogger("uvicorn.error")

CHATTERBOX_URL = f"http://{os_getenv('CHATTER_HOST', 'localhost')}:{os_getenv('CHATTER_PORT', '8000')}/speech"
MAX_INPUT_LENGTH = 500
VOICES_FILE = os_getenv("VOICES_FILE", "voices.yml")
DEFAULT_CONTENT_TYPE = "application/octet-stream"
SILENCE_MS = 0  # Configurable silence between chunks in milliseconds


class VoicePreset(BaseModel):
    """Voice preset configuration."""

    name: str
    audio_prompt: str | None = Field(default=None)
    exaggeration: float = Field(default=0.5, ge=0.0, le=1.0)
    cfg: float = Field(default=0.5, ge=0.0, le=1.0)
    temperature: float | None = Field(default=None, ge=0.0, le=1.0)
    filter_chain: str | None = Field(default=None, description="Audio filter chain to apply")


class TTSRequest(BaseModel):
    """OpenAI-compatible TTS request model.

    This is what your app sends to the proxy. The model field is ignored (we always use Chatterbox),
    but we keep it to maintain compatibility with OpenAI's API.
    """

    model: str | None = "tts-1"
    voice: str
    input: str


def load_voice_presets() -> dict[str, VoicePreset]:
    """Load and validate voice presets from a YAML configuration file.

    This function reads voice preset configurations from a YAML file, ensuring each preset
    has valid settings and properly formatted audio prompt paths. It handles both local
    and reference directory-based audio prompts.

    Returns:
        Dictionary mapping voice IDs to their corresponding VoicePreset configurations.

    Raises:
        FileNotFoundError: If the voices configuration file cannot be found.
        YAMLError: If the YAML file contains invalid syntax or structure.
    """
    try:
        with Path(VOICES_FILE).open(encoding="utf-8") as f:
            raw_presets = yaml_safe_load(f)

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
    except YAMLError:
        logger.exception("Invalid YAML in voices file")
        raise
    else:
        return presets


# Load voice presets at startup
try:
    VOICE_PRESETS = load_voice_presets()
except (FileNotFoundError, YAMLError, ValueError):
    logger.exception("Failed to load voice presets")
    VOICE_PRESETS = {}  # Fallback to empty dict if loading fails


@app.get("/health")
async def health_check() -> dict[str, str]:
    """Check the health status of the TTS service.

    This endpoint provides a simple health check mechanism to verify the service is
    running and ready to handle requests. It's useful for monitoring and load balancing.

    Returns:
        Dictionary containing the service status.
    """
    return {"status": "healthy"}


@app.get("/v1/audio/voices")
async def list_voices() -> list[dict[str, str]]:
    """List all available voice presets in OpenAI-compatible format.

    This endpoint provides a list of available voice presets in the format expected by
    OpenAI's TTS API clients. Each voice is represented by an ID and display name.

    Returns:
        List of dictionaries, each containing:
        - id: Unique identifier for the voice preset
        - name: Human-readable name of the voice
    """
    return [{"id": k, "name": v.name} for k, v in VOICE_PRESETS.items()]


@app.get("/v1/audio/models")
async def list_models() -> list[dict[str, str | int]]:
    """List available TTS models in OpenAI-compatible format.

    This endpoint maintains compatibility with OpenAI's API by providing model information
    in the expected format. While we always use Chatterbox under the hood, this endpoint
    allows clients to check available models without modification.

    Returns:
        List containing a single dictionary with:
        - id: Model identifier ("chatterbox")
        - created: Timestamp of model creation
    """
    return [{"id": "chatterbox", "created": 1713956629}]


def split_text_at_punctuation(text: str, max_length: int) -> list[str]:
    """Split text into chunks at natural speech break points.

    This function intelligently splits long text into smaller chunks that respect natural
    speech patterns. It prioritises splitting at sentence boundaries (punctuation followed
    by whitespace) to maintain natural speech flow. If no suitable punctuation is found,
    it falls back to splitting at word boundaries.

    The function ensures each chunk is no longer than the specified maximum length while
    preserving the natural rhythm of speech.

    Args:
        text: The text to be split into chunks.
        max_length: Maximum allowed length for each chunk.

    Returns:
        List of text chunks, each no longer than max_length, split at natural break points.
    """
    if len(text) <= max_length:
        return [text]

    # Find the last punctuation mark followed by whitespace within max_length
    pattern = r"[.!?]\s"
    matches = list(re_finditer(pattern, text[:max_length]))

    if not matches:
        # If no punctuation found, split at the last space
        last_space = text[:max_length].rfind(" ")
        split_point = last_space + 1 if last_space > 0 else max_length
    else:
        # Use the last punctuation mark found
        split_point = matches[-1].end()

    # Split and trim
    first_chunk = text[:split_point].strip()
    remaining_text = text[split_point:].strip()

    # Recursively process remaining text
    return [first_chunk, *split_text_at_punctuation(remaining_text, max_length)]


async def generate_audio_chunk(text: str, preset: VoicePreset) -> bytes:
    """Generate audio for a single text chunk using Chatterbox TTS.

    This function handles the conversion of a text chunk to speech using the specified
    voice preset. It constructs the appropriate payload for the Chatterbox API, including
    voice cloning settings and emotional parameters.

    Args:
        text: The text to convert to speech.
        preset: Voice preset configuration containing settings for voice cloning,
               emotional intensity, and generation quality.

    Returns:
        Raw audio data as bytes.

    Raises:
        HTTPException: If audio generation fails or returns invalid data
    """
    start_time = perf_counter()

    payload = {
        "text": text,
        "exaggeration": preset.exaggeration,
        "cfg": preset.cfg,
    }

    if preset.audio_prompt:
        payload["audio_prompt"] = preset.audio_prompt
    elif preset.temperature is not None:
        payload["temperature"] = preset.temperature

    try:
        async with HTTPXAsyncClient(timeout=30.0) as client:
            response = await client.post(CHATTERBOX_URL, json=payload)
            response.raise_for_status()

            # Log the content type and first few bytes to help identify the format
            content_type = response.headers.get("content-type", "unknown")
            logger.debug("Received audio with content-type: %s", content_type)

            # Check for common audio format signatures
            data = response.content
            if len(data) >= 4:
                header = data[:4]
                if header.startswith(b"RIFF"):
                    logger.debug("Detected WAV format (RIFF header)")
                elif header.startswith(b"OggS"):
                    logger.debug("Detected OGG format (OggS header)")
                elif header.startswith(b"fLaC"):
                    logger.debug("Detected FLAC format (fLaC header)")
                elif header.startswith(b"ID3"):
                    logger.debug("Detected MP3 format (ID3 header)")
                else:
                    logger.warning("Unknown audio format, header bytes: %r", header)

            # Validate the audio data
            try:
                AudioSegment.from_wav(BytesIO(data))
            except CouldntDecodeError as e:
                logger.exception("Invalid audio data received")
                raise HTTPException(
                    status_code=500, detail="Invalid audio data received from TTS service"
                ) from e

            generation_time = perf_counter() - start_time
            logger.info("Generated audio chunk: size=%d bytes, time=%.2fs", len(data), generation_time)

            return data

    except Exception as e:
        logger.exception("Failed to generate audio chunk")
        raise HTTPException(status_code=500, detail=f"Failed to generate audio: {e!s}") from e


async def get_wav_parameters(audio_data: bytes) -> tuple[int, int, int]:
    """Extract WAV parameters from the audio data.

    Args:
        audio_data: Raw WAV data

    Returns:
        Tuple of (sample_rate, channels, bitrate)

    Raises:
        HTTPException: If audio data is invalid
    """
    try:
        audio = AudioSegment.from_wav(BytesIO(audio_data))
        return audio.frame_rate, audio.channels, audio.frame_width * 8
    except CouldntDecodeError as e:
        logger.exception("Failed to parse WAV data")
        raise HTTPException(status_code=500, detail="Invalid WAV data received") from e


async def generate_silence(duration_ms: int, sample_rate: int, channels: int) -> bytes:
    """Generate WAV-encoded silence.

    Args:
        duration_ms: Duration in milliseconds
        sample_rate: Sample rate in Hz
        channels: Number of audio channels

    Returns:
        WAV-encoded silence as bytes

    Raises:
        HTTPException: If silence generation fails
    """
    try:
        # Use the same sample rate as the input audio
        silence = AudioSegment.silent(duration=duration_ms, frame_rate=sample_rate)
        if channels == 1:
            silence = silence.set_channels(1)

        output = BytesIO()
        silence.export(output, format="wav")
        output.seek(0)
        return output.read()
    except Exception as e:
        logger.exception("Failed to generate silence")
        raise HTTPException(status_code=500, detail="Failed to generate silence") from e


async def process_audio_chunk(
    audio_data: bytes,
    filter_chain: str | None,
) -> bytes:
    """Process an audio chunk with the specified filter chain.

    Args:
        audio_data: Raw WAV data
        filter_chain: Name of the filter chain to apply

    Returns:
        Processed audio data as MP3 bytes

    Raises:
        HTTPException: If audio processing fails
    """
    try:
        # Convert to AudioSegment for processing
        audio = AudioSegment.from_wav(BytesIO(audio_data))

        # Log audio parameters
        logger.debug(
            "Processing audio: %dHz, %d channels, %d bits/sample",
            audio.frame_rate,
            audio.channels,
            audio.sample_width * 8,
        )

        # Apply filter chain if specified
        if filter_chain and filter_chain in FILTER_CHAINS:
            audio = FILTER_CHAINS[filter_chain].apply(audio)

        # Export as MP3 with appropriate quality settings
        # Note: bitrate is per-channel, so we double it for mono to get desired total bitrate
        output = BytesIO()
        audio.export(
            output,
            format="mp3",
            bitrate="128k",  # Will be 64k effective for mono
            parameters=["-q:a", "4"],  # Slightly lower quality VBR for speech
        )
        output.seek(0)
        result = output.read()

        # Log compression stats
        original_size = len(audio_data)
        compressed_size = len(result)
        logger.info(
            "Audio compression: %d bytes -> %d bytes (%.1f%%)",
            original_size,
            compressed_size,
            (compressed_size / original_size) * 100,
        )
    except Exception as e:
        logger.exception("Failed to process audio chunk")
        raise HTTPException(status_code=500, detail="Failed to process audio") from e
    else:
        return result


async def stream_audio_chunks(
    text_chunks: list[str],
    preset: VoicePreset,
) -> AsyncGenerator[bytes]:
    """Stream audio chunks with proper MP3 frame handling.

    Args:
        text_chunks: List of text chunks to process
        preset: Voice preset configuration

    Yields:
        MP3 audio data chunks

    Raises:
        HTTPException: If audio streaming fails
    """
    start_time = perf_counter()
    total_bytes = 0

    try:
        # Get parameters from first chunk
        first_chunk = await generate_audio_chunk(text_chunks[0], preset)
        first_chunk = await process_audio_chunk(first_chunk, preset.filter_chain)
        total_bytes += len(first_chunk)

        # Get audio parameters from the first chunk
        audio = AudioSegment.from_mp3(BytesIO(first_chunk))
        logger.info(
            "Audio parameters: %dHz, %d channels, %d bits/sample",
            audio.frame_rate,
            audio.channels,
            audio.sample_width * 8,
        )

        # Yield first chunk
        yield first_chunk

        # Process remaining chunks
        for chunk in text_chunks[1:]:
            # Add silence between chunks if needed
            if SILENCE_MS > 0:
                # Generate silence with matching parameters
                wav_silence = await generate_silence(SILENCE_MS, audio.frame_rate, audio.channels)
                silence = await process_audio_chunk(wav_silence, None)
                total_bytes += len(silence)
                yield silence

            # Generate and yield next chunk
            audio_data = await generate_audio_chunk(chunk, preset)
            audio_data = await process_audio_chunk(audio_data, preset.filter_chain)
            total_bytes += len(audio_data)
            yield audio_data

        total_time = perf_counter() - start_time
        logger.info(
            "Completed audio stream: chunks=%d, total_bytes=%d, time=%.2fs",
            len(text_chunks),
            total_bytes,
            total_time,
        )

    except Exception as e:
        logger.exception("Error during audio streaming")
        raise HTTPException(status_code=500, detail=f"Error during audio streaming: {e!s}") from e


@app.post("/v1/audio/speech")
async def openai_tts(req: TTSRequest) -> StreamingResponse:
    """Generate speech from text using Chatterbox TTS with OpenAI-compatible interface.

    This endpoint provides a drop-in replacement for OpenAI's TTS API, handling text of any
    length by automatically splitting it into manageable chunks. It maintains natural speech
    patterns by splitting at sentence boundaries and streams the resulting audio segments
    directly to the client without re-encoding.

    Args:
        req: TTS request containing the text to speak and voice selection.

    Returns:
        Streaming response containing the generated audio as an MP3 file.

    Raises:
        HTTPException: 400 if voice preset unknown, 500 if audio generation fails
    """
    logger.info(
        "Received TTS request: model=%s, voice=%s, input_length=%d",
        req.model,
        req.voice,
        len(req.input),
    )

    if req.voice not in VOICE_PRESETS:
        logger.warning("Unknown voice preset: %s", req.voice)
        raise HTTPException(status_code=400, detail=f"Unknown voice preset: {req.voice}")

    preset = VOICE_PRESETS[req.voice]

    # Split text into chunks if needed
    text_chunks = split_text_at_punctuation(req.input, MAX_INPUT_LENGTH)
    logger.info("Split text into %d chunks", len(text_chunks))

    return StreamingResponse(
        stream_audio_chunks(text_chunks, preset),
        media_type="audio/mpeg",
    )
