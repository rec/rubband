from __future__ import annotations

import importlib
import sys
from typing import Protocol, cast

import numpy as np
from numpy.typing import NDArray


class _RubbandBackend(Protocol):
    def stretch_float32(
        self,
        audio: NDArray[np.float32],
        sample_rate: int,
        time_ratio: float,
        pitch_scale: float,
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


def stretch_float32(
    audio: NDArray[np.float32],
    sample_rate: int,
    time_ratio: float,
    pitch_scale: float,
) -> NDArray[np.float32]:
    planar = np.asarray(audio, dtype=np.float32, order="F")
    result = _rubband.stretch_float32(
        planar,
        sample_rate,
        time_ratio,
        pitch_scale,
    )
    return np.asarray(result, dtype=np.float32, order="C")
