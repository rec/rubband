from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

class Stretcher:
    def __init__(
        self,
        sample_rate: int,
        channels: int,
        time_ratio: float,
        pitch_scale: float,
        option_flags: int,
    ) -> None: ...
    def study(self, audio: NDArray[np.float32], final: bool = False) -> None: ...
    def process(self, audio: NDArray[np.float32], final: bool = False) -> None: ...
    def set_time_ratio(self, ratio: float) -> None: ...
    def set_pitch_scale(self, scale: float) -> None: ...
    def available(self) -> int: ...
    def retrieve(self) -> NDArray[np.float32]: ...

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
