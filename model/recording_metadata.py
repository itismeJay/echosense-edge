"""Portable filename metadata for multilingual microphone recordings."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re


DELIVERIES = ("normal", "angry", "mocking", "quiet", "fast")
NOISE_CONDITIONS = ("quiet", "classroom")
DISTANCES = ("near", "far", "media")
_SAFE_ID = re.compile(r"^[A-Za-z0-9_-]+$")


def _validated_identifier(value, field):
    normalized = str(value or "").strip()
    if (
        not normalized
        or "__" in normalized
        or _SAFE_ID.fullmatch(normalized) is None
    ):
        raise ValueError(f"invalid {field}")
    return normalized


@dataclass(frozen=True, slots=True)
class RecordingDescriptor:
    sentence_id: str
    speaker_id: str
    delivery: str
    noise: str
    distance: str

    def __post_init__(self):
        object.__setattr__(
            self,
            "sentence_id",
            _validated_identifier(self.sentence_id, "sentence_id"),
        )
        object.__setattr__(
            self,
            "speaker_id",
            _validated_identifier(self.speaker_id, "speaker_id"),
        )
        if self.delivery not in DELIVERIES:
            raise ValueError("invalid delivery")
        if self.noise not in NOISE_CONDITIONS:
            raise ValueError("invalid noise")
        if self.distance not in DISTANCES:
            raise ValueError("invalid distance")

    def filename(self):
        return (
            f"{self.sentence_id}__{self.speaker_id}__{self.delivery}__"
            f"{self.noise}__{self.distance}.wav"
        )

    @classmethod
    def from_filename(cls, value):
        stem = Path(value).stem
        parts = stem.split("__")
        if len(parts) == 2:
            sentence_id, distance = parts
            return cls(
                sentence_id=sentence_id,
                speaker_id="legacy",
                delivery="normal",
                noise="quiet",
                distance=distance,
            )
        if len(parts) != 5:
            raise ValueError("recording filename does not contain valid metadata")
        return cls(
            sentence_id=parts[0],
            speaker_id=parts[1],
            delivery=parts[2],
            noise=parts[3],
            distance=parts[4],
        )
