"""Small, non-audio payload previews derived from synchronized samples."""

import numpy as np

from audio.audio_event import float32_samples_to_int16


def get_waveform_snapshot(audio_np, num_points=40):
    if len(audio_np) == 0:
        return [0] * num_points
    indices = np.linspace(0, len(audio_np) - 1, num_points, dtype=int)
    return [abs(int(audio_np[index])) for index in indices]


def get_waveform_snapshot_float32(samples, num_points=40):
    """Build the legacy 40-point waveform from synchronized event samples."""
    return get_waveform_snapshot(
        float32_samples_to_int16(samples),
        num_points=num_points,
    )
