"""Audio processing filters for voice profiles.

This module provides audio processing chains for different voice profiles,
allowing real-time transformation of TTS output to match specific character voices.
"""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from typing import TYPE_CHECKING

import soundfile as sf
from pedalboard import Chorus, Gain, Pedalboard, PitchShift, Reverb, load_plugin
from pydub import AudioSegment

if TYPE_CHECKING:
    from collections.abc import Callable


@dataclass
class FilterChain:
    """Configuration for a chain of audio filters."""

    name: str
    apply: Callable[[AudioSegment], AudioSegment]


def apply_glados_filters(audio: AudioSegment) -> AudioSegment:
    """Apply GLaDOS-style processing to the audio using VST plugins.

    This implements a processing chain that adds the characteristic GLaDOS sound:
    1. TAL-Vocoder for the core robotic sound
    2. Graillon3 for hard-tune and formant shifting
    3. Subtle pitch shift to brighten
    4. Light chorus and reverb for presence

    Args:
        audio: Input audio segment

    Returns:
        Processed audio segment
    """
    # Convert to mono if needed
    if audio.channels > 1:
        audio = audio.set_channels(1)

    # Export to WAV for processing
    wav_io = BytesIO()
    audio.export(wav_io, format="wav")
    wav_io.seek(0)

    # Load the audio with soundfile
    voice, sr = sf.read(wav_io)

    # Create the processing chain
    board = Pedalboard([
        load_plugin("/app/plugins/TAL-Vocoder.vst3"),  # 11-band vocoder
        load_plugin("/app/plugins/Graillon3.vst3"),  # hard-tune + formant shift
        PitchShift(semitones=+3.0),  # brighten the core
        Gain(gain_db=-9),  # keep headroom
        Chorus(rate_hz=0.3, depth=0.05),  # subtle movement
        Reverb(room_size=0.25),  # electronic presence
    ])

    # Process the audio
    processed = board(voice, sr)

    # Convert back to AudioSegment
    output_io = BytesIO()
    sf.write(output_io, processed, sr, format="WAV")
    output_io.seek(0)
    return AudioSegment.from_wav(output_io)


# Define available filter chains
FILTER_CHAINS: dict[str, FilterChain] = {
    "glados": FilterChain(
        name="GLaDOS",
        apply=apply_glados_filters,
    ),
    # Add more filter chains here as needed
}
