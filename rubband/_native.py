from __future__ import annotations

import importlib
import sys
from functools import cache
from typing import Protocol, cast


class _NativeStretcher(Protocol):
    def study(self, audio: object, final: bool) -> None: ...

    def process(self, audio: object, final: bool) -> None: ...

    def reset(self) -> None: ...

    def set_time_ratio(self, ratio: float) -> None: ...

    def set_pitch_scale(self, scale: float) -> None: ...

    def set_formant_scale(self, scale: float) -> None: ...

    def set_transients_option(self, options: int) -> None: ...

    def set_detector_option(self, options: int) -> None: ...

    def set_phase_option(self, options: int) -> None: ...

    def set_formant_option(self, options: int) -> None: ...

    def set_pitch_option(self, options: int) -> None: ...

    def get_time_ratio(self) -> float: ...

    def get_pitch_scale(self) -> float: ...

    def get_formant_scale(self) -> float: ...

    def get_preferred_start_pad(self) -> int: ...

    def get_start_delay(self) -> int: ...

    def get_latency(self) -> int: ...

    def get_channel_count(self) -> int: ...

    def set_expected_input_duration(self, samples: int) -> None: ...

    def set_max_process_size(self, samples: int) -> None: ...

    def get_process_size_limit(self) -> int: ...

    def get_samples_required(self) -> int: ...

    def available(self) -> int: ...

    def retrieve(self) -> object: ...


class _RubbandBackend(Protocol):
    Stretcher: type[_NativeStretcher]

    def metadata(
        self,
        sample_rate: int,
        channels: int,
        time_ratio: float,
        pitch_scale: float,
        option_flags: int,
    ) -> dict[str, int | float]: ...

    def stretch_float32(
        self,
        audio: object,
        sample_rate: int,
        time_ratio: float,
        pitch_scale: float,
        option_flags: int,
    ) -> object: ...


def _load_backend() -> _RubbandBackend:
    try:
        return cast(_RubbandBackend, importlib.import_module("rubband._rubband"))
    except (ImportError, OSError) as e:
        sys.exit(_backend_load_error(e))


def _backend_load_error(error: BaseException) -> str:
    return (
        "Rubband could not load its native Rubber Band extension. "
        "Install librubberband and reinstall rubband. "
        f"Original error: {error}"
    )


@cache
def _backend() -> _RubbandBackend:
    return _load_backend()


class Stretcher:
    def __init__(
        self,
        sample_rate: int,
        channels: int,
        time_ratio: float,
        pitch_scale: float,
        option_flags: int,
    ) -> None:
        self.handle = _backend().Stretcher(
            sample_rate,
            channels,
            time_ratio,
            pitch_scale,
            option_flags,
        )

    def study(self, audio: object, final: bool) -> None:
        self.handle.study(audio, final)

    def process(self, audio: object, final: bool) -> None:
        self.handle.process(audio, final)

    def reset(self) -> None:
        self.handle.reset()

    def set_time_ratio(self, ratio: float) -> None:
        self.handle.set_time_ratio(ratio)

    def set_pitch_scale(self, scale: float) -> None:
        self.handle.set_pitch_scale(scale)

    def set_formant_scale(self, scale: float) -> None:
        self.handle.set_formant_scale(scale)

    def set_transients_option(self, options: int) -> None:
        self.handle.set_transients_option(options)

    def set_detector_option(self, options: int) -> None:
        self.handle.set_detector_option(options)

    def set_phase_option(self, options: int) -> None:
        self.handle.set_phase_option(options)

    def set_formant_option(self, options: int) -> None:
        self.handle.set_formant_option(options)

    def set_pitch_option(self, options: int) -> None:
        self.handle.set_pitch_option(options)

    def get_time_ratio(self) -> float:
        return self.handle.get_time_ratio()

    def get_pitch_scale(self) -> float:
        return self.handle.get_pitch_scale()

    def get_formant_scale(self) -> float:
        return self.handle.get_formant_scale()

    def get_preferred_start_pad(self) -> int:
        return self.handle.get_preferred_start_pad()

    def get_start_delay(self) -> int:
        return self.handle.get_start_delay()

    def get_latency(self) -> int:
        return self.handle.get_latency()

    def get_channel_count(self) -> int:
        return self.handle.get_channel_count()

    def set_expected_input_duration(self, samples: int) -> None:
        self.handle.set_expected_input_duration(samples)

    def set_max_process_size(self, samples: int) -> None:
        self.handle.set_max_process_size(samples)

    def get_process_size_limit(self) -> int:
        return self.handle.get_process_size_limit()

    def get_samples_required(self) -> int:
        return self.handle.get_samples_required()

    def available(self) -> int:
        return self.handle.available()

    def retrieve(self) -> object:
        return self.handle.retrieve()


def metadata(
    sample_rate: int,
    channels: int,
    time_ratio: float,
    pitch_scale: float,
    option_flags: int,
) -> dict[str, int | float]:
    return _backend().metadata(
        sample_rate,
        channels,
        time_ratio,
        pitch_scale,
        option_flags,
    )


def stretch_float32(
    audio: object,
    sample_rate: int,
    time_ratio: float,
    pitch_scale: float,
    option_flags: int,
) -> object:
    return _backend().stretch_float32(
        audio,
        sample_rate,
        time_ratio,
        pitch_scale,
        option_flags,
    )
