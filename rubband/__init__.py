from __future__ import annotations

import enum
import math
import sys
from collections.abc import Callable, Mapping
from enum import StrEnum
from functools import cached_property
from threading import Lock

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
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


class LiveWindowOption(StrEnum):
    short = enum.auto()
    medium = enum.auto()


class LiveFormantOption(StrEnum):
    shifted = enum.auto()
    preserved = enum.auto()


class LiveChannelsOption(StrEnum):
    apart = enum.auto()
    together = enum.auto()


class LivePresetOption(StrEnum):
    default = enum.auto()


class Options(BaseModel):
    """Rubber Band option flags for constructing a stretcher.

    Time and pitch ratios are not options in Rubber Band itself. Pass them to
    `stretch()` or to `Stretcher` as `initial_time_ratio` and
    `initial_pitch_scale`.

    Attributes:
        preset: Rubber Band preset to apply before explicit option values.
        process: Offline or real-time processing mode.
        stretch: Time-stretching algorithm behavior.
        transients: Transient preservation behavior.
        detector: Transient detector tuned for the source material.
        phase: Phase handling behavior.
        threading: Rubber Band internal threading behavior.
        window: Analysis window size.
        smoothing: Whether Rubber Band smoothing is enabled.
        formant: Formant handling while pitch shifting.
        pitch: Pitch-shifting quality and consistency preference.
        channels: Whether channels are processed independently or together.
        engine: Rubber Band engine selection.
    """

    model_config = ConfigDict(frozen=True)

    preset: PresetOption = Field(
        default=PresetOption.default,
        description="Rubber Band preset to apply before explicit option values.",
    )
    process: ProcessOption = Field(
        default=ProcessOption.offline,
        description="Offline or real-time processing mode.",
    )
    stretch: StretchOption = Field(
        default=StretchOption.elastic,
        description="Time-stretching algorithm behavior.",
    )
    transients: TransientsOption = Field(
        default=TransientsOption.crisp,
        description="Transient preservation behavior.",
    )
    detector: DetectorOption = Field(
        default=DetectorOption.compound,
        description="Transient detector tuned for the source material.",
    )
    phase: PhaseOption = Field(
        default=PhaseOption.laminar,
        description="Phase handling behavior.",
    )
    threading: ThreadingOption = Field(
        default=ThreadingOption.never,
        description="Rubber Band internal threading behavior.",
    )
    window: WindowOption = Field(
        default=WindowOption.standard,
        description="Analysis window size.",
    )
    smoothing: SmoothingOption = Field(
        default=SmoothingOption.off,
        description="Whether Rubber Band smoothing is enabled.",
    )
    formant: FormantOption = Field(
        default=FormantOption.shifted,
        description="Formant handling while pitch shifting.",
    )
    pitch: PitchOption = Field(
        default=PitchOption.high_quality,
        description="Pitch-shifting quality and consistency preference.",
    )
    channels: ChannelsOption = Field(
        default=ChannelsOption.together,
        description="Whether channels are processed independently or together.",
    )
    engine: EngineOption = Field(
        default=EngineOption.finer,
        description="Rubber Band engine selection.",
    )

    @property
    def option_flags(self) -> int:
        """Combined Rubber Band option bitmask for the current options."""
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


class LiveOptions(BaseModel):
    """Rubber Band LiveShifter option flags.

    Attributes:
        preset: Rubber Band live preset to apply before explicit option values.
        window: Live shifter window size.
        formant: Formant handling while pitch shifting.
        channels: Whether channels are processed independently or together.
    """

    model_config = ConfigDict(frozen=True)

    preset: LivePresetOption = Field(
        default=LivePresetOption.default,
        description="Rubber Band live preset to apply before explicit option values.",
    )
    window: LiveWindowOption = Field(
        default=LiveWindowOption.short,
        description="Live shifter window size.",
    )
    formant: LiveFormantOption = Field(
        default=LiveFormantOption.shifted,
        description="Formant handling while pitch shifting.",
    )
    channels: LiveChannelsOption = Field(
        default=LiveChannelsOption.apart,
        description="Whether channels are processed independently or together.",
    )

    @property
    def option_flags(self) -> int:
        """Combined Rubber Band LiveShifter option bitmask."""
        return (
            _LIVE_PRESET_OPTIONS[self.preset]
            | _LIVE_WINDOW_OPTIONS[self.window]
            | _LIVE_FORMANT_OPTIONS[self.formant]
            | _LIVE_CHANNELS_OPTIONS[self.channels]
        )


class AudioBuffer(BaseModel):
    """Output audio returned by `stretch()` and `Stretcher.retrieve()`.

    Attributes:
        data: C-contiguous float32 audio exposed through the Python buffer protocol.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    data: object = Field(description="Underlying C-contiguous float32 audio buffer.")

    @field_validator("data", mode="before")
    @classmethod
    def validate_data(cls, value: object) -> object:
        """Validate and normalize an output audio buffer.

        Args:
            value: Object exposing a C-contiguous float32 buffer.

        Returns:
            A memoryview over the provided audio.
        """
        view = memoryview(_validate_result(value, None))  # ty: ignore[invalid-argument-type]
        if view.ndim == 2 and view.shape is not None and view.shape[1] == 0:
            raise ValueError("audio must contain at least one channel")
        return view

    @property
    def dtype(self) -> str:
        """Return the public dtype name for this buffer."""
        return "float32"

    @property
    def frames(self) -> int:
        """Return the number of audio frames."""
        view = memoryview(self.data)  # ty: ignore[invalid-argument-type]
        if view.shape is None:
            raise ValueError("audio buffer has no shape")
        return view.shape[0]

    @property
    def channels(self) -> int:
        """Return the number of audio channels."""
        view = memoryview(self.data)  # ty: ignore[invalid-argument-type]
        if view.shape is None:
            raise ValueError("audio buffer has no shape")
        if view.ndim == 1:
            return 1
        return view.shape[1]

    @property
    def shape(self) -> tuple[int] | tuple[int, int]:
        """Return the audio shape as ``(frames,)`` or ``(frames, channels)``."""
        view = memoryview(self.data)  # ty: ignore[invalid-argument-type]
        if view.shape is None:
            raise ValueError("audio buffer has no shape")
        if view.ndim == 1:
            return (view.shape[0],)
        return (view.shape[0], view.shape[1])

    def memoryview(self) -> object:
        """Return the underlying zero-copy memoryview."""
        return memoryview(self.data)  # ty: ignore[invalid-argument-type]

    def numpy(self) -> object:
        """Return a NumPy view of the audio buffer.

        Returns:
            NumPy array view over the same audio memory.
        """
        try:
            import numpy as np
        except ImportError as error:
            raise ImportError(
                "NumPy is required to convert AudioBuffer to NumPy"
            ) from error
        return np.asarray(self.data)

    def torch(self) -> object:
        """Return a PyTorch CPU tensor view of the audio buffer.

        Returns:
            PyTorch tensor view over the same audio memory.
        """
        try:
            import torch
        except ImportError as error:
            raise ImportError(
                "PyTorch is required to convert AudioBuffer to PyTorch"
            ) from error
        return torch.frombuffer(self.data, dtype=torch.float32).reshape(self.shape)


class RubberBandMetadata(BaseModel):
    """Read-only values reported by a configured Rubber Band stretcher.

    Attributes:
        engine_version: Rubber Band engine version.
        available: Initial available output frame count.
        preferred_start_pad: Input frames Rubber Band recommends prepending.
        start_delay: Output frames to discard after start padding.
        time_ratio: Configured output-to-input duration ratio.
        pitch_scale: Configured output-to-input pitch ratio.
    """

    model_config = ConfigDict(frozen=True)

    engine_version: int = Field(description="Rubber Band engine version.")
    available: int = Field(description="Initial available output frame count.")
    preferred_start_pad: int = Field(
        description="Input frames Rubber Band recommends prepending before processing."
    )
    start_delay: int = Field(
        description="Output frames to discard after start padding."
    )
    time_ratio: float = Field(description="Configured output-to-input duration ratio.")
    pitch_scale: float = Field(description="Configured output-to-input pitch ratio.")


class Stretcher(BaseModel):
    """Stateful Rubber Band stretcher for offline or real-time processing.

    Args:
        sample_rate: Input sample rate in Hz.
        channels: Number of audio channels.
        options: Rubber Band option flags used to construct the stretcher.
        initial_time_ratio: Initial output-to-input duration ratio.
        initial_pitch_scale: Initial output-to-input pitch ratio.
        logger: Optional callback for Rubber Band debug log messages.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    _lock: Lock = PrivateAttr(default_factory=Lock)
    _started: bool = PrivateAttr(default=False)
    _logger: Callable[..., object] | None = PrivateAttr(default=None)

    sample_rate: int = Field(description="Input sample rate in Hz.")
    channels: int = Field(description="Number of audio channels.")
    options: Options = Field(
        default_factory=Options,
        description="Rubber Band option flags used to construct the stretcher.",
    )
    initial_time_ratio: float = Field(
        default=1.0,
        description="Initial output-to-input duration ratio.",
    )
    initial_pitch_scale: float = Field(
        default=1.0,
        description="Initial output-to-input pitch ratio.",
    )

    def __init__(
        self,
        sample_rate: int,
        channels: int,
        options: Options | None = None,
        initial_time_ratio: float = 1.0,
        initial_pitch_scale: float = 1.0,
        logger: Callable[..., object] | None = None,
    ) -> None:
        super().__init__(
            sample_rate=sample_rate,
            channels=channels,
            options=options or Options(),
            initial_time_ratio=initial_time_ratio,
            initial_pitch_scale=initial_pitch_scale,
        )
        self._logger = _validate_logger(logger)

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
            self._logger,
        )

    def study(self, audio: object, final: bool = False) -> None:
        """Analyze audio before offline processing.

        Args:
            audio: Contiguous CPU float32 audio with shape ``(frames,)`` or
                ``(frames, channels)``.
            final: Whether this is the last audio block to study.
        """
        normalized = _validate_stretcher_audio(audio, self.channels)
        with self._lock:
            self.native.study(normalized, final)
            self._started = True

    def process(self, audio: object, final: bool = False) -> None:
        """Process audio and make output available through `retrieve()`.

        Args:
            audio: Contiguous CPU float32 audio with shape ``(frames,)`` or
                ``(frames, channels)``.
            final: Whether this is the last audio block to process.
        """
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
        """Set the stretched-to-unstretched duration ratio.

        Args:
            ratio: Positive output-to-input duration ratio.
        """
        ratio = _validate_positive_ratio(ratio)
        with self._lock:
            self._validate_dynamic_ratio_change()
            self.native.set_time_ratio(ratio)

    def set_pitch_scale(self, scale: float) -> None:
        """Set the target-to-source frequency ratio.

        Args:
            scale: Positive output-to-input pitch ratio.
        """
        scale = _validate_positive_ratio(scale)
        with self._lock:
            self._validate_dynamic_ratio_change()
            self.native.set_pitch_scale(scale)

    def set_formant_scale(self, scale: float) -> None:
        """Set Rubber Band's formant scale.

        Args:
            scale: Non-negative formant scale.
        """
        scale = _validate_non_negative_ratio(scale)
        with self._lock:
            self.native.set_formant_scale(scale)

    def set_transients_option(self, option: TransientsOption) -> None:
        """Set transient handling for a real-time stretcher.

        Args:
            option: Transient preservation behavior.
        """
        self._validate_real_time_option_change("transients option")
        with self._lock:
            self.native.set_transients_option(_TRANSIENTS_OPTIONS[option])

    def set_detector_option(self, option: DetectorOption) -> None:
        """Set transient detector behavior for a real-time stretcher.

        Args:
            option: Transient detector tuned for the source material.
        """
        self._validate_real_time_option_change("detector option")
        with self._lock:
            self.native.set_detector_option(_DETECTOR_OPTIONS[option])

    def set_phase_option(self, option: PhaseOption) -> None:
        """Set phase handling behavior.

        Args:
            option: Phase handling behavior.
        """
        with self._lock:
            self.native.set_phase_option(_PHASE_OPTIONS[option])

    def set_formant_option(self, option: FormantOption) -> None:
        """Set formant handling behavior.

        Args:
            option: Formant handling while pitch shifting.
        """
        with self._lock:
            self.native.set_formant_option(_FORMANT_OPTIONS[option])

    def set_pitch_option(self, option: PitchOption) -> None:
        """Set pitch processing behavior for a real-time stretcher.

        Args:
            option: Pitch-shifting quality and consistency preference.
        """
        self._validate_real_time_option_change("pitch option")
        with self._lock:
            self.native.set_pitch_option(_PITCH_OPTIONS[option])

    def get_time_ratio(self) -> float:
        """Return the current stretched-to-unstretched duration ratio."""
        with self._lock:
            return self.native.get_time_ratio()

    def get_pitch_scale(self) -> float:
        """Return the current target-to-source pitch ratio."""
        with self._lock:
            return self.native.get_pitch_scale()

    def get_formant_scale(self) -> float:
        """Return the current Rubber Band formant scale."""
        with self._lock:
            return self.native.get_formant_scale()

    def get_preferred_start_pad(self) -> int:
        """Return the input frames Rubber Band recommends prepending."""
        with self._lock:
            return self.native.get_preferred_start_pad()

    def get_start_delay(self) -> int:
        """Return the output frames to discard after start padding."""
        with self._lock:
            return self.native.get_start_delay()

    def get_latency(self) -> int:
        """Return the processing latency in frames."""
        with self._lock:
            return self.native.get_latency()

    def get_channel_count(self) -> int:
        """Return the configured channel count."""
        with self._lock:
            return self.native.get_channel_count()

    def set_expected_input_duration(self, samples: int) -> None:
        """Set the expected input duration for offline processing.

        Args:
            samples: Non-negative expected input frame count.
        """
        samples = _validate_sample_count(samples)
        with self._lock:
            self.native.set_expected_input_duration(samples)

    def set_max_process_size(self, samples: int) -> None:
        """Set the maximum block size accepted by `process()`.

        Args:
            samples: Non-negative maximum input frame count per process call.
        """
        samples = _validate_sample_count(samples)
        with self._lock:
            if self._started:
                raise ValueError(
                    "max process size cannot be changed after study or process"
                )
            self.native.set_max_process_size(samples)

    def get_process_size_limit(self) -> int:
        """Return the current maximum process block size in frames."""
        with self._lock:
            return self.native.get_process_size_limit()

    def get_samples_required(self) -> int:
        """Return the input frames required before more output can be produced."""
        with self._lock:
            return self.native.get_samples_required()

    def available(self) -> int:
        """Return the number of output frames currently available."""
        with self._lock:
            return self.native.available()

    def retrieve(self) -> AudioBuffer:
        """Return available output audio.

        Returns:
            AudioBuffer with shape ``(frames, channels)``.
        """
        with self._lock:
            result = self.native.retrieve()
        _validate_result(result, self.channels)
        return AudioBuffer(data=result)

    def get_engine_version(self) -> int:
        """Return the active Rubber Band engine version."""
        with self._lock:
            return self.native.get_engine_version()

    def set_key_frame_map(self, key_frames: Mapping[int, int]) -> None:
        """Set a pre-planned offline source-to-target frame mapping.

        Args:
            key_frames: Mapping from source frame positions to target frame positions.
        """
        mapped = _validate_key_frame_map(key_frames)
        with self._lock:
            if self._started:
                raise ValueError(
                    "key frame map cannot be changed after processing starts"
                )
            self.native.set_key_frame_map(mapped)

    def get_frequency_cutoff(self, n: int) -> float:
        """Return Rubber Band's frequency cutoff for the given band index.

        Args:
            n: Frequency cutoff index.
        """
        n = _validate_index(n)
        with self._lock:
            return self.native.get_frequency_cutoff(n)

    def set_frequency_cutoff(self, n: int, frequency: float) -> None:
        """Set Rubber Band's frequency cutoff for the given band index.

        Args:
            n: Frequency cutoff index.
            frequency: Non-negative finite cutoff frequency.
        """
        n = _validate_index(n)
        frequency = _validate_non_negative_ratio(frequency)
        with self._lock:
            self.native.set_frequency_cutoff(n, frequency)

    def get_input_increment(self) -> int:
        """Return Rubber Band's internal input increment."""
        with self._lock:
            return self.native.get_input_increment()

    def get_output_increments(self) -> list[int]:
        """Return accumulated internal output increments."""
        with self._lock:
            return list(self.native.get_output_increments())

    def get_phase_reset_curve(self) -> list[float]:
        """Return accumulated phase reset curve points."""
        with self._lock:
            return list(self.native.get_phase_reset_curve())

    def get_exact_time_points(self) -> list[int]:
        """Return accumulated exact time points."""
        with self._lock:
            return list(self.native.get_exact_time_points())

    def set_debug_level(self, level: int) -> None:
        """Set Rubber Band debug output level for this stretcher.

        Args:
            level: Non-negative Rubber Band debug level.
        """
        level = _validate_debug_level(level)
        with self._lock:
            self.native.set_debug_level(level)

    @staticmethod
    def set_default_debug_level(level: int) -> None:
        """Set the Rubber Band default debug level for future stretchers.

        Args:
            level: Non-negative Rubber Band debug level.
        """
        _native.Stretcher.set_default_debug_level(_validate_debug_level(level))

    def _validate_dynamic_ratio_change(self) -> None:
        if self.options.process == ProcessOption.offline and self._started:
            raise ValueError(
                "offline stretchers cannot change ratios after study or process"
            )

    def _validate_real_time_option_change(self, name: str) -> None:
        if self.options.process == ProcessOption.offline:
            raise ValueError(f"{name} cannot be changed in offline mode")


class LiveShifter(BaseModel):
    """Rubber Band 4.x live pitch shifter.

    Args:
        sample_rate: Input sample rate in Hz.
        channels: Number of audio channels.
        options: Rubber Band LiveShifter option flags.
        logger: Optional callback for Rubber Band debug log messages.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    _lock: Lock = PrivateAttr(default_factory=Lock)
    _logger: Callable[..., object] | None = PrivateAttr(default=None)

    sample_rate: int = Field(description="Input sample rate in Hz.")
    channels: int = Field(description="Number of audio channels.")
    options: LiveOptions = Field(
        default_factory=LiveOptions,
        description="Rubber Band LiveShifter option flags.",
    )

    def __init__(
        self,
        sample_rate: int,
        channels: int,
        options: LiveOptions | None = None,
        logger: Callable[..., object] | None = None,
    ) -> None:
        super().__init__(
            sample_rate=sample_rate,
            channels=channels,
            options=options or LiveOptions(),
        )
        self._logger = _validate_logger(logger)

    @field_validator("sample_rate")
    @classmethod
    def validate_sample_rate(cls, value: int) -> int:
        return _validate_sample_rate(value)

    @field_validator("channels")
    @classmethod
    def validate_channels(cls, value: int) -> int:
        return Stretcher.validate_channels(value)

    @cached_property
    def native(self) -> _native.LiveShifter:
        return _native.LiveShifter(
            self.sample_rate,
            self.channels,
            self.options.option_flags,
            self._logger,
        )

    def reset(self) -> None:
        """Reset internal buffers while retaining current pitch ratio."""
        with self._lock:
            self.native.reset()

    def set_pitch_scale(self, scale: float) -> None:
        """Set the target-to-source frequency ratio.

        Args:
            scale: Positive output-to-input pitch ratio.
        """
        scale = _validate_positive_ratio(scale)
        with self._lock:
            self.native.set_pitch_scale(scale)

    def set_formant_scale(self, scale: float) -> None:
        """Set Rubber Band's live formant scale.

        Args:
            scale: Non-negative formant scale.
        """
        scale = _validate_non_negative_ratio(scale)
        with self._lock:
            self.native.set_formant_scale(scale)

    def get_pitch_scale(self) -> float:
        """Return the current target-to-source pitch ratio."""
        with self._lock:
            return self.native.get_pitch_scale()

    def get_formant_scale(self) -> float:
        """Return the current Rubber Band formant scale."""
        with self._lock:
            return self.native.get_formant_scale()

    def get_start_delay(self) -> int:
        """Return the output frames to discard for time alignment."""
        with self._lock:
            return self.native.get_start_delay()

    def get_channel_count(self) -> int:
        """Return the configured channel count."""
        with self._lock:
            return self.native.get_channel_count()

    def set_formant_option(self, option: LiveFormantOption) -> None:
        """Set live formant handling behavior.

        Args:
            option: Live formant handling while pitch shifting.
        """
        with self._lock:
            self.native.set_formant_option(_LIVE_FORMANT_OPTIONS[option])

    def get_block_size(self) -> int:
        """Return the exact frame count required by each `shift()` call."""
        with self._lock:
            return self.native.get_block_size()

    def shift(self, audio: object) -> AudioBuffer:
        """Pitch-shift one fixed-size live audio block.

        Args:
            audio: Contiguous CPU float32 audio with exactly `get_block_size()`
                frames and the configured channel count.

        Returns:
            AudioBuffer with the same shape as the input block.
        """
        normalized = _validate_live_audio(audio, self.channels, self.get_block_size())
        with self._lock:
            result = self.native.shift(normalized)
        _validate_result(result, self.channels)
        return AudioBuffer(data=result)

    def shift_into(self, audio: object, output: object) -> None:
        """Pitch-shift one live audio block into caller-provided output memory.

        Args:
            audio: Contiguous CPU float32 input audio with exactly
                `get_block_size()` frames and the configured channel count.
            output: Writable contiguous CPU float32 output audio with matching shape.
        """
        block_size = self.get_block_size()
        normalized = _validate_live_audio(audio, self.channels, block_size)
        normalized_output = _validate_live_audio(output, self.channels, block_size)
        with self._lock:
            self.native.shift_into(normalized, normalized_output)

    def set_debug_level(self, level: int) -> None:
        """Set Rubber Band debug output level for this live shifter.

        Args:
            level: Non-negative Rubber Band debug level.
        """
        level = _validate_debug_level(level)
        with self._lock:
            self.native.set_debug_level(level)

    @staticmethod
    def set_default_debug_level(level: int) -> None:
        """Set the Rubber Band default debug level for future live shifters.

        Args:
            level: Non-negative Rubber Band debug level.
        """
        _native.LiveShifter.set_default_debug_level(_validate_debug_level(level))


def stretch(
    audio: object,
    sample_rate: int,
    time_ratio: float = 1.0,
    pitch_scale: float = 1.0,
    options: Options | None = None,
) -> AudioBuffer:
    """Stretch and pitch-shift CPU float32 audio in one offline call.

    Input must be contiguous CPU float32 audio with shape ``(frames,)`` for
    mono or ``(frames, channels)`` for multichannel audio. Objects may expose
    DLPack or the Python buffer protocol.

    Args:
        audio: Contiguous CPU float32 input audio.
        sample_rate: Input sample rate in Hz.
        time_ratio: Output-to-input duration ratio.
        pitch_scale: Output-to-input pitch ratio.
        options: Rubber Band option flags for the one-shot stretcher.

    Returns:
        AudioBuffer containing C-contiguous float32 output.
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
    return AudioBuffer(data=result)


def metadata(
    sample_rate: int,
    channels: int = 1,
    time_ratio: float = 1.0,
    pitch_scale: float = 1.0,
    options: Options | None = None,
) -> RubberBandMetadata:
    """Return metadata for a Rubber Band stretcher configuration.

    Args:
        sample_rate: Input sample rate in Hz.
        channels: Number of audio channels.
        time_ratio: Output-to-input duration ratio.
        pitch_scale: Output-to-input pitch ratio.
        options: Rubber Band option flags for the metadata query.

    Returns:
        Read-only metadata for the configured stretcher.
    """
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


def _validate_index(value: int) -> int:
    if isinstance(value, bool):
        raise TypeError("index must be an integer")
    if value < 0:
        raise ValueError("index must be non-negative")
    return value


def _validate_debug_level(value: int) -> int:
    if isinstance(value, bool):
        raise TypeError("debug level must be an integer")
    if value < 0:
        raise ValueError("debug level must be non-negative")
    return value


def _validate_logger(
    value: Callable[..., object] | None,
) -> Callable[..., object] | None:
    if value is not None and not callable(value):
        raise TypeError("logger must be callable")
    return value


def _validate_key_frame_map(key_frames: Mapping[int, int]) -> dict[int, int]:
    if not isinstance(key_frames, Mapping):
        raise TypeError("key frame map must be a mapping")
    mapped: dict[int, int] = {}
    for source, target in key_frames.items():
        mapped[_validate_sample_count(source)] = _validate_sample_count(target)
    return mapped


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


def _validate_result(result: object, channels: int | None) -> object:
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
    return view


def _validate_stretcher_audio(audio: object, channels: int) -> object:
    normalized, detected_channels = _validate_audio(audio)
    if detected_channels is not None and detected_channels != channels:
        raise ValueError("audio channel count does not match stretcher")
    return normalized


def _validate_live_audio(audio: object, channels: int, frames: int) -> object:
    normalized, detected_channels = _validate_audio(audio)
    if detected_channels is not None and detected_channels != channels:
        raise ValueError("audio channel count does not match live shifter")
    try:
        view = memoryview(normalized)  # ty: ignore[invalid-argument-type]
    except TypeError:
        return normalized
    if view.shape is None:
        raise ValueError("audio must have shape (frames,) or (frames, channels)")
    if view.shape[0] != frames:
        raise ValueError("audio frame count must match live shifter block size")
    return normalized


_CONSTANTS = _native.option_constants()

_PRESET_OPTIONS = {
    PresetOption.default: _CONSTANTS["preset_default"],
    PresetOption.percussive: _CONSTANTS["preset_percussive"],
}
_PROCESS_OPTIONS = {
    ProcessOption.offline: _CONSTANTS["process_offline"],
    ProcessOption.real_time: _CONSTANTS["process_real_time"],
}
_STRETCH_OPTIONS = {
    StretchOption.elastic: _CONSTANTS["stretch_elastic"],
    StretchOption.precise: _CONSTANTS["stretch_precise"],
}
_TRANSIENTS_OPTIONS = {
    TransientsOption.crisp: _CONSTANTS["transients_crisp"],
    TransientsOption.mixed: _CONSTANTS["transients_mixed"],
    TransientsOption.smooth: _CONSTANTS["transients_smooth"],
}
_DETECTOR_OPTIONS = {
    DetectorOption.compound: _CONSTANTS["detector_compound"],
    DetectorOption.percussive: _CONSTANTS["detector_percussive"],
    DetectorOption.soft: _CONSTANTS["detector_soft"],
}
_PHASE_OPTIONS = {
    PhaseOption.laminar: _CONSTANTS["phase_laminar"],
    PhaseOption.independent: _CONSTANTS["phase_independent"],
}
_THREADING_OPTIONS = {
    ThreadingOption.auto: _CONSTANTS["threading_auto"],
    ThreadingOption.never: _CONSTANTS["threading_never"],
    ThreadingOption.always: _CONSTANTS["threading_always"],
}
_WINDOW_OPTIONS = {
    WindowOption.standard: _CONSTANTS["window_standard"],
    WindowOption.short: _CONSTANTS["window_short"],
    WindowOption.long: _CONSTANTS["window_long"],
}
_SMOOTHING_OPTIONS = {
    SmoothingOption.off: _CONSTANTS["smoothing_off"],
    SmoothingOption.on: _CONSTANTS["smoothing_on"],
}
_FORMANT_OPTIONS = {
    FormantOption.shifted: _CONSTANTS["formant_shifted"],
    FormantOption.preserved: _CONSTANTS["formant_preserved"],
}
_PITCH_OPTIONS = {
    PitchOption.high_speed: _CONSTANTS["pitch_high_speed"],
    PitchOption.high_quality: _CONSTANTS["pitch_high_quality"],
    PitchOption.high_consistency: _CONSTANTS["pitch_high_consistency"],
}
_CHANNELS_OPTIONS = {
    ChannelsOption.apart: _CONSTANTS["channels_apart"],
    ChannelsOption.together: _CONSTANTS["channels_together"],
}
_ENGINE_OPTIONS = {
    EngineOption.faster: _CONSTANTS["engine_faster"],
    EngineOption.finer: _CONSTANTS["engine_finer"],
}
_LIVE_PRESET_OPTIONS = {
    LivePresetOption.default: _CONSTANTS["live_preset_default"],
}
_LIVE_WINDOW_OPTIONS = {
    LiveWindowOption.short: _CONSTANTS["live_window_short"],
    LiveWindowOption.medium: _CONSTANTS["live_window_medium"],
}
_LIVE_FORMANT_OPTIONS = {
    LiveFormantOption.shifted: _CONSTANTS["live_formant_shifted"],
    LiveFormantOption.preserved: _CONSTANTS["live_formant_preserved"],
}
_LIVE_CHANNELS_OPTIONS = {
    LiveChannelsOption.apart: _CONSTANTS["live_channels_apart"],
    LiveChannelsOption.together: _CONSTANTS["live_channels_together"],
}
