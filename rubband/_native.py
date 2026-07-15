from __future__ import annotations

import importlib
import sys
from typing import Protocol, cast

import numpy as np
from numpy.typing import NDArray


class _NativeStretcher(Protocol):
    def study(self, audio: NDArray[np.float32], final: bool) -> None: ...

    def process(self, audio: NDArray[np.float32], final: bool) -> None: ...

    def available(self) -> int: ...

    def retrieve(self) -> NDArray[np.float32]: ...


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
        audio: NDArray[np.float32],
        sample_rate: int,
        time_ratio: float,
        pitch_scale: float,
        option_flags: int,
    ) -> NDArray[np.float32]: ...


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


_rubband = _load_backend()


class Stretcher:
    def __init__(
        self,
        sample_rate: int,
        channels: int,
        time_ratio: float,
        pitch_scale: float,
        option_flags: int,
    ) -> None:
        self.handle = _rubband.Stretcher(
            sample_rate,
            channels,
            time_ratio,
            pitch_scale,
            option_flags,
        )

    def study(self, audio: NDArray[np.float32], final: bool) -> None:
        self.handle.study(np.asarray(audio, dtype=np.float32, order="F"), final)

    def process(self, audio: NDArray[np.float32], final: bool) -> None:
        self.handle.process(np.asarray(audio, dtype=np.float32, order="F"), final)

    def available(self) -> int:
        return self.handle.available()

    def retrieve(self) -> NDArray[np.float32]:
        return np.asarray(self.handle.retrieve(), dtype=np.float32, order="C")


def metadata(
    sample_rate: int,
    channels: int,
    time_ratio: float,
    pitch_scale: float,
    option_flags: int,
) -> dict[str, int | float]:
    return _rubband.metadata(
        sample_rate,
        channels,
        time_ratio,
        pitch_scale,
        option_flags,
    )


def stretch_float32(
    audio: NDArray[np.float32],
    sample_rate: int,
    time_ratio: float,
    pitch_scale: float,
    option_flags: int,
) -> NDArray[np.float32]:
    planar = np.asarray(audio, dtype=np.float32, order="F")
    result = _rubband.stretch_float32(
        planar,
        sample_rate,
        time_ratio,
        pitch_scale,
        option_flags,
    )
    return np.asarray(result, dtype=np.float32, order="C")
