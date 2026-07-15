from __future__ import annotations

import enum
import math
import sys
from enum import StrEnum
from functools import cached_property
from threading import Lock

from pydantic import (
    BaseModel,
    ConfigDict,
    PrivateAttr,
    field_validator,
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


class Options(BaseModel):
    """Rubber Band option flags for constructing a stretcher.

    Time and pitch ratios are not options in Rubber Band itself. Pass them to
    `stretch()` or to `Stretcher` as `initial_time_ratio` and
    `initial_pitch_scale`.
    """

    model_config = ConfigDict(frozen=True)

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
    """Read-only values reported by a configured Rubber Band stretcher."""

    model_config = ConfigDict(frozen=True)

    engine_version: int
    available: int
    preferred_start_pad: int
    start_delay: int
    time_ratio: float
    pitch_scale: float


class Stretcher(BaseModel):
    """Stateful Rubber Band stretcher for offline or real-time processing."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    _lock: Lock = PrivateAttr(default_factory=Lock)
    _started: bool = PrivateAttr(default=False)

    sample_rate: int
    channels: int
    options: Options = Options()
    initial_time_ratio: float = 1.0
    initial_pitch_scale: float = 1.0

    def __init__(
        self,
        sample_rate: int,
        channels: int,
        options: Options | None = None,
        initial_time_ratio: float = 1.0,
        initial_pitch_scale: float = 1.0,
    ) -> None:
        super().__init__(
            sample_rate=sample_rate,
            channels=channels,
            options=options or Options(),
            initial_time_ratio=initial_time_ratio,
            initial_pitch_scale=initial_pitch_scale,
        )

    @field_validator("sample_rate")
    @classmethod
    def validate_sample_rate(cls, value: int) -> int:
        return _validate_sample_rate(value)

    @field_validator("channels")
    @classmethod
    def validate_channels(cls, value: int) -> int:
        if isinstance(value, bool):
            raise TypeError("channels must be an integer")
        if value < 1 or value > 256:
            raise ValueError("channels must be between 1 and 256")
        return value

    @field_validator("initial_time_ratio", "initial_pitch_scale")
    @classmethod
    def validate_initial_ratio(cls, value: float) -> float:
        return _validate_positive_ratio(value)

    @cached_property
    def native(self) -> _native.Stretcher:
        return _native.Stretcher(
            self.sample_rate,
            self.channels,
            self.initial_time_ratio,
            self.initial_pitch_scale,
            self.options.option_flags,
        )

    def study(self, audio: object, final: bool = False) -> None:
        normalized = _validate_stretcher_audio(audio, self.channels)
        with self._lock:
            self.native.study(normalized, final)
            self._started = True

    def process(self, audio: object, final: bool = False) -> None:
        normalized = _validate_stretcher_audio(audio, self.channels)
        with self._lock:
            self.native.process(normalized, final)
            self._started = True

    def reset(self) -> None:
        """Reset Rubber Band's internal buffers while retaining current ratios."""
        with self._lock:
            self.native.reset()
            self._started = False

    def set_time_ratio(self, ratio: float) -> None:
        """Set stretched-to-unstretched duration ratio."""
        ratio = _validate_positive_ratio(ratio)
        with self._lock:
            self._validate_dynamic_ratio_change()
            self.native.set_time_ratio(ratio)

    def set_pitch_scale(self, scale: float) -> None:
        """Set target-to-source frequency ratio."""
        scale = _validate_positive_ratio(scale)
        with self._lock:
            self._validate_dynamic_ratio_change()
            self.native.set_pitch_scale(scale)

    def set_formant_scale(self, scale: float) -> None:
        scale = _validate_non_negative_ratio(scale)
        with self._lock:
            self.native.set_formant_scale(scale)

    def set_transients_option(self, option: TransientsOption) -> None:
        self._validate_real_time_option_change("transients option")
        with self._lock:
            self.native.set_transients_option(_TRANSIENTS_OPTIONS[option])

    def set_detector_option(self, option: DetectorOption) -> None:
        self._validate_real_time_option_change("detector option")
        with self._lock:
            self.native.set_detector_option(_DETECTOR_OPTIONS[option])

    def set_phase_option(self, option: PhaseOption) -> None:
        with self._lock:
            self.native.set_phase_option(_PHASE_OPTIONS[option])

    def set_formant_option(self, option: FormantOption) -> None:
        with self._lock:
            self.native.set_formant_option(_FORMANT_OPTIONS[option])

    def set_pitch_option(self, option: PitchOption) -> None:
        self._validate_real_time_option_change("pitch option")
        with self._lock:
            self.native.set_pitch_option(_PITCH_OPTIONS[option])

    def get_time_ratio(self) -> float:
        with self._lock:
            return self.native.get_time_ratio()

    def get_pitch_scale(self) -> float:
        with self._lock:
            return self.native.get_pitch_scale()

    def get_formant_scale(self) -> float:
        with self._lock:
            return self.native.get_formant_scale()

    def get_preferred_start_pad(self) -> int:
        with self._lock:
            return self.native.get_preferred_start_pad()

    def get_start_delay(self) -> int:
        with self._lock:
            return self.native.get_start_delay()

    def get_latency(self) -> int:
        with self._lock:
            return self.native.get_latency()

    def get_channel_count(self) -> int:
        with self._lock:
            return self.native.get_channel_count()

    def set_expected_input_duration(self, samples: int) -> None:
        samples = _validate_sample_count(samples)
        with self._lock:
            self.native.set_expected_input_duration(samples)

    def set_max_process_size(self, samples: int) -> None:
        samples = _validate_sample_count(samples)
        with self._lock:
            if self._started:
                raise ValueError(
                    "max process size cannot be changed after study or process"
                )
            self.native.set_max_process_size(samples)

    def get_process_size_limit(self) -> int:
        with self._lock:
            return self.native.get_process_size_limit()

    def get_samples_required(self) -> int:
        with self._lock:
            return self.native.get_samples_required()

    def available(self) -> int:
        with self._lock:
            return self.native.available()

    def retrieve(self) -> object:
        with self._lock:
            result = self.native.retrieve()
        _validate_result(result, self.channels)
        return result

    def _validate_dynamic_ratio_change(self) -> None:
        if self.options.process == ProcessOption.offline and self._started:
            raise ValueError(
                "offline stretchers cannot change ratios after study or process"
            )

    def _validate_real_time_option_change(self, name: str) -> None:
        if self.options.process == ProcessOption.offline:
            raise ValueError(f"{name} cannot be changed in offline mode")


def stretch(
    audio: object,
    sample_rate: int,
    time_ratio: float = 1.0,
    pitch_scale: float = 1.0,
    options: Options | None = None,
) -> object:
    """Stretch and pitch-shift CPU float32 audio in one offline call.

    Input must be contiguous CPU float32 audio with shape ``(frames,)`` for
    mono or ``(frames, channels)`` for multichannel audio. Objects may expose
    DLPack or the Python buffer protocol.
    """
    sample_rate = _validate_sample_rate(sample_rate)
    time_ratio = _validate_positive_ratio(time_ratio)
    pitch_scale = _validate_positive_ratio(pitch_scale)
    resolved_options = options or Options()
    normalized, channels = _validate_audio(audio)
    result = _native.stretch_float32(
        normalized,
        sample_rate,
        time_ratio,
        pitch_scale,
        resolved_options.option_flags,
    )
    _validate_result(result, channels)
    return result


def metadata(
    sample_rate: int,
    channels: int = 1,
    time_ratio: float = 1.0,
    pitch_scale: float = 1.0,
    options: Options | None = None,
) -> RubberBandMetadata:
    """Return metadata for a Rubber Band stretcher configuration."""
    sample_rate = _validate_sample_rate(sample_rate)
    time_ratio = _validate_positive_ratio(time_ratio)
    pitch_scale = _validate_positive_ratio(pitch_scale)
    resolved_options = options or Options()
    Stretcher.validate_channels(channels)
    return RubberBandMetadata.model_validate(
        _native.metadata(
            sample_rate,
            channels,
            time_ratio,
            pitch_scale,
            resolved_options.option_flags,
        )
    )


def main() -> None:
    sys.exit("rubband does not provide a command line interface yet.")


def _validate_sample_rate(value: int) -> int:
    if isinstance(value, bool):
        raise TypeError("sample_rate must be an integer")
    if value < 8000 or value > 192000:
        raise ValueError("sample_rate must be between 8000 and 192000")
    return value


def _validate_positive_ratio(value: float) -> float:
    if not math.isfinite(value) or value <= 0:
        raise ValueError("ratio must be finite and greater than zero")
    return value


def _validate_non_negative_ratio(value: float) -> float:
    if not math.isfinite(value) or value < 0:
        raise ValueError("ratio must be finite and non-negative")
    return value


def _validate_sample_count(value: int) -> int:
    if isinstance(value, bool):
        raise TypeError("sample count must be an integer")
    if value < 0:
        raise ValueError("sample count must be non-negative")
    return value


def _validate_audio(audio: object) -> tuple[object, int | None]:
    try:
        view = memoryview(audio)  # ty: ignore[invalid-argument-type]
    except TypeError as error:
        if hasattr(audio, "__dlpack__"):
            return audio, None
        raise TypeError(
            "audio must expose DLPack or the Python buffer protocol"
        ) from error
    if view.shape is None:
        raise ValueError("audio must have shape (frames,) or (frames, channels)")
    shape = view.shape
    if view.format not in {"f", "<f", "=f"} or view.itemsize != 4:
        raise TypeError("audio must have dtype float32")
    if view.ndim == 1:
        if shape[0] == 0:
            raise ValueError("audio must contain at least one frame")
        if not view.c_contiguous:
            raise ValueError("audio must be C-contiguous")
        return audio, 1
    if view.ndim != 2:
        raise ValueError("audio must have shape (frames,) or (frames, channels)")
    if shape[0] == 0:
        raise ValueError("audio must contain at least one frame")
    if shape[1] == 0:
        raise ValueError("audio must contain at least one channel")
    if not view.c_contiguous:
        raise ValueError("audio must be C-contiguous")
    return audio, shape[1]


def _validate_result(result: object, channels: int | None) -> None:
    try:
        view = memoryview(result)  # ty: ignore[invalid-argument-type]
    except TypeError as error:
        raise TypeError("native backend returned non-buffer audio") from error
    if view.shape is None:
        raise ValueError("native backend returned audio with the wrong rank")
    shape = view.shape
    if view.format not in {"f", "<f", "=f"} or view.itemsize != 4:
        raise TypeError("native backend returned non-float32 audio")
    if view.ndim not in {1, 2}:
        raise ValueError("native backend returned audio with the wrong rank")
    result_channels = 1 if view.ndim == 1 else shape[1]
    if channels is not None and result_channels != channels:
        raise ValueError("native backend returned audio with the wrong channel count")
    if not view.c_contiguous:
        raise ValueError("native backend returned non-contiguous audio")


def _validate_stretcher_audio(audio: object, channels: int) -> object:
    normalized, detected_channels = _validate_audio(audio)
    if detected_channels is not None and detected_channels != channels:
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
