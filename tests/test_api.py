from __future__ import annotations

import io
import math
import wave

import numpy as np
import pytest
from numpy.typing import NDArray

import rubband
from rubband import _native

SAMPLE_RATE = 48_000


def test_stretch_accepts_one_second_mono_audio(
    monkeypatch: pytest.MonkeyPatch,
    file_regression: object,
) -> None:
    audio = sine_wave(seconds=1.0)

    def stretch_float32(
        backend_audio: NDArray[np.float32],
        sample_rate: int,
        time_ratio: float,
        pitch_scale: float,
    ) -> NDArray[np.float32]:
        assert backend_audio.shape == (SAMPLE_RATE, 1)
        assert sample_rate == SAMPLE_RATE
        assert time_ratio == 1.25
        assert pitch_scale == 0.5
        return backend_audio.copy()

    monkeypatch.setattr(_native, "stretch_float32", stretch_float32)

    result = rubband.stretch(
        audio,
        rubband.StretchOptions(
            sample_rate=SAMPLE_RATE,
            time_ratio=1.25,
            pitch_scale=0.5,
        ),
    )

    assert result.shape == (SAMPLE_RATE,)
    file_regression.check(
        wav_bytes(result),
        extension=".wav",
        binary=True,
    )


def test_stretch_accepts_one_second_stereo_audio(
    monkeypatch: pytest.MonkeyPatch,
    file_regression: object,
) -> None:
    audio = np.column_stack((sine_wave(seconds=1.0), sine_wave(seconds=1.0, hz=660)))

    def stretch_float32(
        backend_audio: NDArray[np.float32],
        sample_rate: int,
        time_ratio: float,
        pitch_scale: float,
    ) -> NDArray[np.float32]:
        assert sample_rate == SAMPLE_RATE
        assert time_ratio == 1.0
        assert pitch_scale == 2.0
        return backend_audio.copy()

    monkeypatch.setattr(_native, "stretch_float32", stretch_float32)

    result = rubband.stretch(
        np.ascontiguousarray(audio, dtype=np.float32),
        rubband.StretchOptions(sample_rate=SAMPLE_RATE, pitch_scale=2.0),
    )

    assert result.shape == (SAMPLE_RATE, 2)
    file_regression.check(
        wav_bytes(result),
        extension=".wav",
        binary=True,
    )


def test_stretch_rejects_non_numpy_audio() -> None:
    with pytest.raises(TypeError, match="NumPy ndarray"):
        rubband.stretch(
            [0.0],  # type: ignore[arg-type]
            rubband.StretchOptions(sample_rate=SAMPLE_RATE),
        )


def test_stretch_rejects_non_float32_audio() -> None:
    audio = np.zeros(SAMPLE_RATE, dtype=np.float64)

    with pytest.raises(TypeError, match="float32"):
        rubband.stretch(
            audio,  # type: ignore[arg-type]
            rubband.StretchOptions(sample_rate=SAMPLE_RATE),
        )


def test_stretch_rejects_non_contiguous_audio() -> None:
    audio = np.zeros((SAMPLE_RATE, 2), dtype=np.float32)[::2]

    with pytest.raises(ValueError, match="C-contiguous"):
        rubband.stretch(audio, rubband.StretchOptions(sample_rate=SAMPLE_RATE))


def test_stretch_rejects_out_of_range_sample_rate() -> None:
    with pytest.raises(ValueError, match="between 8000 and 192000"):
        rubband.StretchOptions(sample_rate=7999)


def test_stretch_rejects_non_positive_ratios() -> None:
    with pytest.raises(ValueError, match="ratio"):
        rubband.StretchOptions(sample_rate=SAMPLE_RATE, time_ratio=0)
    with pytest.raises(ValueError, match="ratio"):
        rubband.StretchOptions(sample_rate=SAMPLE_RATE, pitch_scale=0)


def test_backend_load_error_is_clear() -> None:
    message = _native._backend_load_error(ImportError("Library not loaded"))

    assert "could not load its native Rubber Band extension" in message
    assert "Install librubberband and reinstall rubband" in message
    assert "Library not loaded" in message


def test_backend_load_failure_exits_without_traceback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_import(name: str) -> object:
        raise ImportError(f"missing {name}")

    monkeypatch.setattr(_native.importlib, "import_module", fail_import)

    with pytest.raises(SystemExit) as error:
        _native._load_backend()

    assert "could not load its native Rubber Band extension" in str(error.value)
    assert "missing rubband._rubband" in str(error.value)


def sine_wave(seconds: float, hz: float = 440.0) -> NDArray[np.float32]:
    frames = int(SAMPLE_RATE * seconds)
    return np.array(
        [math.sin(2.0 * math.pi * hz * i / SAMPLE_RATE) * 0.25 for i in range(frames)],
        dtype=np.float32,
    )


def wav_bytes(audio: NDArray[np.float32]) -> bytes:
    if audio.ndim == 1:
        framed = audio.reshape(audio.shape[0], 1)
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
