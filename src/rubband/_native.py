from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from . import _rubband


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
