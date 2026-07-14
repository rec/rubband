from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

def stretch_float32(
    audio: NDArray[np.float32],
    sample_rate: int,
    time_ratio: float,
    pitch_scale: float,
) -> NDArray[np.float32]: ...
