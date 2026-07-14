from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

def metadata(
    sample_rate: int,
    channels: int,
    time_ratio: float,
    pitch_scale: float,
    option_flags: int,
) -> dict[str, int | float]: ...
def stretch_float32(
    audio: NDArray[np.float32],
    sample_rate: int,
    time_ratio: float,
    pitch_scale: float,
    option_flags: int,
) -> NDArray[np.float32]: ...
