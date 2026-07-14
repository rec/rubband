from __future__ import annotations

import enum
import sys
from enum import StrEnum
from functools import cached_property
from threading import Lock
from typing import Self, cast

import numpy as np
from numpy.typing import NDArray
from pydantic import (
    BaseModel,
    ConfigDict,
    PrivateAttr,
    field_validator,
    model_validator,
)

from . import _native


class ProcessOption(StrEnum):
    offline = enum.auto()
    real_time = enum.auto()


class StretchOption(StrEnum):
    elastic = enum.auto()
    precise = enum.auto()


class TransientsOption(StrEnum):
    crisp = enum.auto()
    mixed = enum.auto()
    smooth = enum.auto()


class DetectorOption(StrEnum):
    compound = enum.auto()
    percussive = enum.auto()
    soft = enum.auto()


class PhaseOption(StrEnum):
    laminar = enum.auto()
    independent = enum.auto()


class ThreadingOption(StrEnum):
    auto = enum.auto()
    never = enum.auto()
    always = enum.auto()


class WindowOption(StrEnum):
    standard = enum.auto()
    short = enum.auto()
    long = enum.auto()


class SmoothingOption(StrEnum):
    off = enum.auto()
    on = enum.auto()


class FormantOption(StrEnum):
    shifted = enum.auto()
    preserved = enum.auto()


class PitchOption(StrEnum):
    high_speed = enum.auto()
    high_quality = enum.auto()
    high_consistency = enum.auto()


class ChannelsOption(StrEnum):
    apart = enum.auto()
    together = enum.auto()


class EngineOption(StrEnum):
    faster = enum.auto()
    finer = enum.auto()


class PresetOption(StrEnum):
    default = enum.auto()
    percussive = enum.auto()


class StretchOptions(BaseModel):
    model_config = ConfigDict(frozen=True)

    sample_rate: int
    time_ratio: float = 1.0
    pitch_scale: float = 1.0
    preset: PresetOption = PresetOption.default
    process: ProcessOption = ProcessOption.offline
    stretch: StretchOption = StretchOption.elastic
    transients: TransientsOption = TransientsOption.crisp
    detector: DetectorOption = DetectorOption.compound
    phase: PhaseOption = PhaseOption.laminar
    threading: ThreadingOption = ThreadingOption.never
    window: WindowOption = WindowOption.standard
    smoothing: SmoothingOption = SmoothingOption.off
    formant: FormantOption = FormantOption.shifted
    pitch: PitchOption = PitchOption.high_quality
    channels: ChannelsOption = ChannelsOption.together
    engine: EngineOption = EngineOption.finer

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

    @property
    def option_flags(self) -> int:
        return (
            _PRESET_OPTIONS[self.preset]
            | _PROCESS_OPTIONS[self.process]
            | _STRETCH_OPTIONS[self.stretch]
            | _TRANSIENTS_OPTIONS[self.transients]
            | _DETECTOR_OPTIONS[self.detector]
            | _PHASE_OPTIONS[self.phase]
            | _THREADING_OPTIONS[self.threading]
            | _WINDOW_OPTIONS[self.window]
            | _SMOOTHING_OPTIONS[self.smoothing]
            | _FORMANT_OPTIONS[self.formant]
            | _PITCH_OPTIONS[self.pitch]
            | _CHANNELS_OPTIONS[self.channels]
            | _ENGINE_OPTIONS[self.engine]
        )


class RubberBandMetadata(BaseModel):
    model_config = ConfigDict(frozen=True)

    engine_version: int
    available: int
    preferred_start_pad: int
    start_delay: int
    time_ratio: float
    pitch_scale: float


class Stretcher(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    _lock: Lock = PrivateAttr(default_factory=Lock)

    sample_rate: int
    channels: int
    options: StretchOptions | None = None

    def __init__(
        self,
        sample_rate: int,
        channels: int,
        options: StretchOptions | None = None,
    ) -> None:
        super().__init__(sample_rate=sample_rate, channels=channels, options=options)

    @field_validator("sample_rate")
    @classmethod
    def validate_sample_rate(cls, value: int) -> int:
        return StretchOptions.validate_sample_rate(value)

    @field_validator("channels")
    @classmethod
    def validate_channels(cls, value: int) -> int:
        if isinstance(value, bool):
            raise TypeError("channels must be an integer")
        if value < 1 or value > 256:
            raise ValueError("channels must be between 1 and 256")
        return value

    @model_validator(mode="after")
    def validate_options(self) -> Self:
        if self.options is not None and self.options.sample_rate != self.sample_rate:
            raise ValueError("options sample_rate must match stretcher sample_rate")
        return self

    @cached_property
    def native(self) -> _native.Stretcher:
        options = self.options or StretchOptions(sample_rate=self.sample_rate)
        return _native.Stretcher(
            self.sample_rate,
            self.channels,
            options.time_ratio,
            options.pitch_scale,
            options.option_flags,
        )

    def study(self, audio: NDArray[np.float32], final: bool = False) -> None:
        normalized = _validate_stretcher_audio(audio, self.channels)
        with self._lock:
            self.native.study(normalized, final)

    def process(self, audio: NDArray[np.float32], final: bool = False) -> None:
        normalized = _validate_stretcher_audio(audio, self.channels)
        with self._lock:
            self.native.process(normalized, final)

    def available(self) -> int:
        with self._lock:
            return self.native.available()

    def retrieve(self) -> NDArray[np.float32]:
        with self._lock:
            result = self.native.retrieve()
        _validate_result(result, self.channels)
        return result


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
        options.option_flags,
    )
    _validate_result(result, normalized.shape[1])
    if mono:
        return result[:, 0].copy()
    return result


def metadata(options: StretchOptions, channels: int = 1) -> RubberBandMetadata:
    return RubberBandMetadata.model_validate(
        _native.metadata(
            options.sample_rate,
            channels,
            options.time_ratio,
            options.pitch_scale,
            options.option_flags,
        )
    )


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


def _validate_stretcher_audio(
    audio: NDArray[np.float32],
    channels: int,
) -> NDArray[np.float32]:
    normalized, _ = _validate_audio(audio)
    if normalized.shape[1] != channels:
        raise ValueError("audio channel count does not match stretcher")
    return normalized


_PRESET_OPTIONS = {
    PresetOption.default: 0x00000000,
    PresetOption.percussive: 0x00102000,
}
_PROCESS_OPTIONS = {
    ProcessOption.offline: 0x00000000,
    ProcessOption.real_time: 0x00000001,
}
_STRETCH_OPTIONS = {
    StretchOption.elastic: 0x00000000,
    StretchOption.precise: 0x00000010,
}
_TRANSIENTS_OPTIONS = {
    TransientsOption.crisp: 0x00000000,
    TransientsOption.mixed: 0x00000100,
    TransientsOption.smooth: 0x00000200,
}
_DETECTOR_OPTIONS = {
    DetectorOption.compound: 0x00000000,
    DetectorOption.percussive: 0x00000400,
    DetectorOption.soft: 0x00000800,
}
_PHASE_OPTIONS = {
    PhaseOption.laminar: 0x00000000,
    PhaseOption.independent: 0x00002000,
}
_THREADING_OPTIONS = {
    ThreadingOption.auto: 0x00000000,
    ThreadingOption.never: 0x00010000,
    ThreadingOption.always: 0x00020000,
}
_WINDOW_OPTIONS = {
    WindowOption.standard: 0x00000000,
    WindowOption.short: 0x00100000,
    WindowOption.long: 0x00200000,
}
_SMOOTHING_OPTIONS = {
    SmoothingOption.off: 0x00000000,
    SmoothingOption.on: 0x00800000,
}
_FORMANT_OPTIONS = {
    FormantOption.shifted: 0x00000000,
    FormantOption.preserved: 0x01000000,
}
_PITCH_OPTIONS = {
    PitchOption.high_speed: 0x00000000,
    PitchOption.high_quality: 0x02000000,
    PitchOption.high_consistency: 0x04000000,
}
_CHANNELS_OPTIONS = {
    ChannelsOption.apart: 0x00000000,
    ChannelsOption.together: 0x10000000,
}
_ENGINE_OPTIONS = {
    EngineOption.faster: 0x00000000,
    EngineOption.finer: 0x20000000,
}
