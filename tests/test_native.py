from __future__ import annotations

import io
import math
import wave
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray
from pytest_regressions.file_regression import FileRegressionFixture

import rubband

SAMPLE_RATE = 48_000
PIANO_LEAD = Path(__file__).parent / "test_native/pianolead.wav"


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
        rubband.StretchOptions(sample_rate=SAMPLE_RATE, time_ratio=1.25),
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
        rubband.StretchOptions(sample_rate=SAMPLE_RATE, pitch_scale=2.0),
    )

    assert result.shape == (SAMPLE_RATE,)
    assert np.max(np.abs(result)) > 0.1
    file_regression.check(
        wav_bytes(result),
        extension=".wav",
        binary=True,
    )


def test_native_stretcher_regression(file_regression: FileRegressionFixture) -> None:
    audio = sine_wave(seconds=1.0)
    stretcher = rubband.Stretcher(SAMPLE_RATE, 1)

    stretcher.study(audio, final=True)
    stretcher.process(audio, final=True)
    result = stretcher.retrieve()

    assert result.shape == (SAMPLE_RATE, 1)
    assert np.max(np.abs(result)) > 0.1
    file_regression.check(
        wav_bytes(result[:, 0]),
        extension=".wav",
        binary=True,
    )


def test_native_stretcher_accepts_dynamic_ratio_setters() -> None:
    stretcher = rubband.Stretcher(
        SAMPLE_RATE,
        1,
        options=rubband.StretchOptions(
            sample_rate=SAMPLE_RATE,
            process=rubband.ProcessOption.real_time,
        ),
    )

    stretcher.set_time_ratio(0.75)
    stretcher.set_pitch_scale(1.5)


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
            rubband.StretchOptions(
                sample_rate=SAMPLE_RATE,
                time_ratio=1.25,
                pitch_scale=2.0,
            ),
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


@pytest.mark.parametrize(
    ("semitones", "basename"),
    [
        (-12, "pianolead_pitch_down_12"),
        (-5, "pianolead_pitch_down_5"),
        (5, "pianolead_pitch_up_5"),
        (12, "pianolead_pitch_up_12"),
    ],
)
@pytest.mark.long
def test_native_pianolead_pitch_regression(
    semitones: int,
    basename: str,
    file_regression: FileRegressionFixture,
) -> None:
    audio, sample_rate = read_wav_float32(PIANO_LEAD)

    result = rubband.stretch(
        audio,
        rubband.StretchOptions(
            sample_rate=sample_rate,
            pitch_scale=2.0 ** (semitones / 12.0),
        ),
    )

    assert result.shape[0] >= sample_rate
    assert result.shape[1] == audio.shape[1]
    assert np.max(np.abs(result)) > 0.1
    file_regression.check(
        wav_bytes(result, sample_rate=sample_rate),
        extension=".wav",
        basename=basename,
        binary=True,
        check_fn=check_wav_prefix,
    )


def sine_wave(seconds: float, hz: float = 440.0) -> NDArray[np.float32]:
    frames = int(SAMPLE_RATE * seconds)
    return np.array(
        [math.sin(2.0 * math.pi * hz * i / SAMPLE_RATE) * 0.25 for i in range(frames)],
        dtype=np.float32,
    )


def read_wav_float32(path: Path) -> tuple[NDArray[np.float32], int]:
    with wave.open(str(path), "rb") as wav:
        channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        sample_rate = wav.getframerate()
        frames = wav.getnframes()
        data = wav.readframes(frames)
    if sample_width != 2:
        raise ValueError("test WAV source must use 16-bit PCM samples")
    if channels < 1:
        raise ValueError("test WAV source must contain at least one channel")
    samples = np.frombuffer(data, dtype="<i2").reshape((frames, channels))
    audio = samples.astype(np.float32) / 32768.0
    if channels == 1:
        return np.ascontiguousarray(audio[:, 0], dtype=np.float32), sample_rate
    return np.ascontiguousarray(audio, dtype=np.float32), sample_rate


def check_wav_prefix(expected_path: Path, obtained_path: Path) -> None:
    expected, expected_sample_rate = read_wav_float32(expected_path)
    obtained, obtained_sample_rate = read_wav_float32(obtained_path)
    prefix_length = min(expected.shape[0], obtained.shape[0])

    assert expected_sample_rate == obtained_sample_rate
    assert expected.ndim == obtained.ndim
    assert expected.shape[1] == obtained.shape[1]
    assert prefix_length >= expected_sample_rate
    np.testing.assert_array_equal(
        expected[:prefix_length],
        obtained[:prefix_length],
    )


def wav_bytes(audio: NDArray[np.float32], sample_rate: int = SAMPLE_RATE) -> bytes:
    if audio.ndim == 1:
        framed = audio.reshape((audio.shape[0], 1))
    else:
        framed = audio
    if framed.shape[0] < sample_rate:
        raise ValueError("test audio must be at least one second long")
    clipped = np.clip(framed, -1.0, 1.0)
    pcm = (clipped * 32767.0).astype("<i2")
    with io.BytesIO() as buffer:
        with wave.open(buffer, "wb") as wav:
            wav.setnchannels(framed.shape[1])
            wav.setsampwidth(2)
            wav.setframerate(sample_rate)
            wav.writeframes(pcm.tobytes())
        return buffer.getvalue()
