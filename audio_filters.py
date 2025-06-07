"""Audio processing filters for voice profiles.

This module provides audio processing chains for different voice profiles,
allowing real-time transformation of TTS output to match specific character voices.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from pydub.effects import compress_dynamic_range, normalize

if TYPE_CHECKING:
    from collections.abc import Callable

    from pydub import AudioSegment


@dataclass
class FilterChain:
    """Configuration for a chain of audio filters."""

    name: str
    apply: Callable[[AudioSegment], AudioSegment]


def apply_glados_reverb(audio: AudioSegment) -> AudioSegment:
    """Apply GLaDOS-style reverb to the audio.

    This implements a simplified version of the reverb settings from the guide:
    - Very short pre-delay (10ms)
    - Low reverberance (15%)
    - High damping (20%)
    - Subtle wet signal (-6dB)

    Args:
        audio: Input audio segment

    Returns:
        Audio segment with reverb applied
    """
    # Create a very short, filtered echo for the reverb
    # The echo is filtered to simulate damping and tone control
    echo = audio - 6  # -6dB wet signal
    echo = echo.low_pass_filter(2000)  # Tone High: 100%
    echo = echo.high_pass_filter(200)  # Tone Low: 70%

    # Apply the echo with a very short delay
    # The short delay and low volume create the intimate room feel
    return audio.overlay(echo, position=10, gain_during_overlay=-15)  # 15% reverberance


def apply_glados_filters(audio: AudioSegment) -> AudioSegment:
    """Apply GLaDOS-style processing to the audio.

    This implements the processing chain from the GLaDOS voice guide:
    1. EQ curve for radio/speaker quality
    2. Compression for synthetic feel
    3. Subtle reverb for electronic presence

    Args:
        audio: Input audio segment

    Returns:
        Processed audio segment
    """
    # Convert to mono if needed
    if audio.channels > 1:
        audio = audio.set_channels(1)

    # Apply EQ curve
    # Note: pydub doesn't have a direct EQ filter, so we use a combination of
    # high-pass and low-pass filters to approximate the curve
    audio = audio.high_pass_filter(60)  # Reduce low-end rumble
    audio = audio.low_pass_filter(2000)  # Control high frequencies

    # Apply compression
    audio = compress_dynamic_range(
        audio,
        threshold=-20,
        ratio=3.0,
        attack=8,
        release=75,
    )

    # Apply the improved reverb
    audio = apply_glados_reverb(audio)

    # Normalize to maintain consistent volume
    return normalize(audio)


# Define available filter chains
FILTER_CHAINS: dict[str, FilterChain] = {
    "glados": FilterChain(
        name="GLaDOS",
        apply=apply_glados_filters,
    ),
    # Add more filter chains here as needed
}
