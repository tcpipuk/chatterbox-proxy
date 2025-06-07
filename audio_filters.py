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

    This implements a very subtle reverb that adds just a hint of electronic presence:
    - Minimal pre-delay (5ms)
    - Very low reverberance (8%)
    - High damping (30%)
    - Very subtle wet signal (-12dB)

    Args:
        audio: Input audio segment

    Returns:
        Audio segment with reverb applied
    """
    # Create a very subtle echo
    echo = audio - 12  # -12dB wet signal

    # Apply minimal filtering to the echo
    echo = echo.low_pass_filter(4000)  # Preserve more of the original character
    echo = echo.high_pass_filter(400)  # Just enough to reduce muddiness

    # Apply the echo with minimal delay and very low volume
    return audio.overlay(echo, position=5, gain_during_overlay=-20)  # Very subtle reverb


def apply_glados_filters(audio: AudioSegment) -> AudioSegment:
    """Apply GLaDOS-style processing to the audio.

    This implements a very subtle processing chain that preserves the original voice
    while adding just enough electronic character to match GLaDOS:
    1. Very gentle EQ for slight radio quality
    2. Light compression for controlled dynamics
    3. Minimal reverb for electronic presence

    Args:
        audio: Input audio segment

    Returns:
        Processed audio segment
    """
    # Convert to mono if needed
    if audio.channels > 1:
        audio = audio.set_channels(1)

    # Apply very gentle EQ
    # We want to preserve most of the original voice character
    audio = audio.high_pass_filter(100)  # Just enough to reduce rumble
    audio = audio.low_pass_filter(8000)  # Preserve most of the high end

    # Apply very light compression
    # Just enough to control dynamics without sounding processed
    audio = compress_dynamic_range(
        audio,
        threshold=-24,  # Only compress the loudest parts
        ratio=1.5,  # Very gentle ratio
        attack=20,  # Slower attack to preserve transients
        release=200,  # Longer release for natural decay
    )

    # Apply the subtle reverb
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
