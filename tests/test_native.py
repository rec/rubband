from __future__ import annotations

import io
import math
import wave

import numpy as np
from numpy.typing import NDArray
from pytest_regressions.file_regression import FileRegressionFixture

import rubband

SAMPLE_RATE = 48_000


def test_native_source_audio_regression(file_regression: FileRegressionFixture) -> None:
    audio = sine_wave(seconds=1.0)

    file_regression.check(
        wav_bytes(audio),
        extension=".wav",
        binary=True,
    )


def test_native_stretch_regression(file_regression: FileRegressionFixture) -> None:
    audio = sine_wave(seconds=1.0)

    result = rubband.stretch(
        audio,
        sample_rate=SAMPLE_RATE,
        time_ratio=1.25,
    )

    assert result.shape == (60_000,)
    assert np.max(np.abs(result)) > 0.1
    file_regression.check(
        wav_bytes(result),
        extension=".wav",
        binary=True,
    )


def test_native_pitch_shift_regression(file_regression: FileRegressionFixture) -> None:
    audio = sine_wave(seconds=1.0, hz=330.0)

    result = rubband.stretch(
        audio,
        sample_rate=SAMPLE_RATE,
        pitch_scale=2.0,
    )

    assert result.shape == (SAMPLE_RATE,)
    assert np.max(np.abs(result)) > 0.1
    file_regression.check(
        wav_bytes(result),
        extension=".wav",
        binary=True,
    )


def test_native_stereo_outputs_have_matching_prefixes() -> None:
    audio = np.ascontiguousarray(
        np.column_stack(
            (
                sine_wave(seconds=1.0, hz=330.0),
                sine_wave(seconds=1.0, hz=660.0),
            )
        ),
        dtype=np.float32,
    )

    outputs = [
        rubband.stretch(
            audio,
            sample_rate=SAMPLE_RATE,
            time_ratio=1.25,
            pitch_scale=2.0,
        )
        for _ in range(8)
    ]
    prefix_length = min(o.shape[0] for o in outputs)

    assert prefix_length >= SAMPLE_RATE
    for output in outputs:
        assert output.shape[1] == 2
        np.testing.assert_array_equal(
            outputs[0][:prefix_length],
            output[:prefix_length],
        )


def sine_wave(seconds: float, hz: float = 440.0) -> NDArray[np.float32]:
    frames = int(SAMPLE_RATE * seconds)
    return np.array(
        [math.sin(2.0 * math.pi * hz * i / SAMPLE_RATE) * 0.25 for i in range(frames)],
        dtype=np.float32,
    )


def wav_bytes(audio: NDArray[np.float32]) -> bytes:
    if audio.ndim == 1:
        framed = audio.reshape((audio.shape[0], 1))
    else:
        framed = audio
    if framed.shape[0] < SAMPLE_RATE:
        raise ValueError("test audio must be at least one second long")
    clipped = np.clip(framed, -1.0, 1.0)
    pcm = (clipped * 32767.0).astype("<i2")
    with io.BytesIO() as buffer:
        with wave.open(buffer, "wb") as wav:
            wav.setnchannels(framed.shape[1])
            wav.setsampwidth(2)
            wav.setframerate(SAMPLE_RATE)
            wav.writeframes(pcm.tobytes())
        return buffer.getvalue()
