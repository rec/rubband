from __future__ import annotations

import sys
from typing import cast

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, field_validator

from . import _native


class StretchOptions(BaseModel):
    model_config = ConfigDict(frozen=True)

    sample_rate: int
    time_ratio: float = 1.0
    pitch_scale: float = 1.0

    @field_validator("sample_rate")
    @classmethod
    def validate_sample_rate(cls, value: int) -> int:
        if isinstance(value, bool):
            raise TypeError("sample_rate must be an integer")
        if value < 8000 or value > 192000:
            raise ValueError("sample_rate must be between 8000 and 192000")
        return value

    @field_validator("time_ratio", "pitch_scale")
    @classmethod
    def validate_positive_ratio(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("ratio must be greater than zero")
        return value


def stretch(
    audio: NDArray[np.float32],
    options: StretchOptions,
) -> NDArray[np.float32]:
    """Stretch and pitch-shift CPU NumPy audio with Rubber Band.

    Input must be float32 NumPy audio with shape ``(frames,)`` for mono or
    ``(frames, channels)`` for multichannel audio. It must be C-contiguous.
    """
    normalized, mono = _validate_audio(audio)
    result = _native.stretch_float32(
        normalized,
        options.sample_rate,
        options.time_ratio,
        options.pitch_scale,
    )
    _validate_result(result, normalized.shape[1])
    if mono:
        return result[:, 0].copy()
    return result


def main() -> None:
    sys.exit("rubband does not provide a command line interface yet.")


def _validate_audio(audio: object) -> tuple[NDArray[np.float32], bool]:
    if not isinstance(audio, np.ndarray):
        raise TypeError("audio must be a NumPy ndarray")
    if audio.dtype != np.float32:
        raise TypeError("audio must have dtype float32")
    typed_audio = cast(NDArray[np.float32], audio)
    if audio.ndim == 1:
        if audio.shape[0] == 0:
            raise ValueError("audio must contain at least one frame")
        if not audio.flags.c_contiguous:
            raise ValueError("audio must be C-contiguous")
        return typed_audio.reshape((audio.shape[0], 1)), True
    if audio.ndim != 2:
        raise ValueError("audio must have shape (frames,) or (frames, channels)")
    if audio.shape[0] == 0:
        raise ValueError("audio must contain at least one frame")
    if audio.shape[1] == 0:
        raise ValueError("audio must contain at least one channel")
    if not audio.flags.c_contiguous:
        raise ValueError("audio must be C-contiguous")
    return typed_audio, False


def _validate_result(result: object, channels: int) -> None:
    if not isinstance(result, np.ndarray):
        raise TypeError("native backend returned a non-NumPy result")
    if result.dtype != np.float32:
        raise TypeError("native backend returned non-float32 audio")
    if result.ndim != 2:
        raise ValueError("native backend returned audio with the wrong rank")
    if result.shape[1] != channels:
        raise ValueError("native backend returned audio with the wrong channel count")
    if not result.flags.c_contiguous:
        raise ValueError("native backend returned non-contiguous audio")
