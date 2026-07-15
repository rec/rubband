from __future__ import annotations

import io
import math
import wave
from array import array
from pathlib import Path
from typing import cast

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


def test_native_stretch_preserves_pitch() -> None:
    audio = sine_wave(seconds=1.0)

    result = cast(
        NDArray[np.float32],
        rubband.stretch(
            audio,
            SAMPLE_RATE,
            time_ratio=1.25,
        ).numpy(),
    )

    assert result.shape == (60_000,)
    assert np.max(np.abs(result)) > 0.1
    assert np.all(np.isfinite(result))
    assert dominant_frequency(result) == pytest.approx(440.0, abs=2.0)


def test_native_pitch_shift_doubles_frequency() -> None:
    audio = sine_wave(seconds=1.0, hz=330.0)

    result = cast(
        NDArray[np.float32],
        rubband.stretch(
            audio,
            SAMPLE_RATE,
            pitch_scale=2.0,
        ).numpy(),
    )

    assert result.shape == (SAMPLE_RATE,)
    assert np.max(np.abs(result)) > 0.1
    assert np.all(np.isfinite(result))
    assert dominant_frequency(result) == pytest.approx(660.0, abs=2.0)


def test_native_accepts_dlpack_audio() -> None:
    audio = sine_wave(seconds=1.0)

    result = cast(
        NDArray[np.float32],
        rubband.stretch(DLPackAudio(audio), SAMPLE_RATE).numpy(),
    )

    assert result.shape == (SAMPLE_RATE,)
    assert np.max(np.abs(result)) > 0.1


def test_native_accepts_array_buffer_audio() -> None:
    audio = array("f", sine_wave(seconds=1.0).tolist())

    result = cast(NDArray[np.float32], rubband.stretch(audio, SAMPLE_RATE).numpy())

    assert result.shape == (SAMPLE_RATE,)
    assert np.max(np.abs(result)) > 0.1


def test_native_accepts_memoryview_audio() -> None:
    audio = memoryview(array("f", sine_wave(seconds=1.0).tolist()))

    result = cast(NDArray[np.float32], rubband.stretch(audio, SAMPLE_RATE).numpy())

    assert result.shape == (SAMPLE_RATE,)
    assert np.max(np.abs(result)) > 0.1


def test_native_accepts_contiguous_torch_cpu_tensor() -> None:
    torch = pytest.importorskip("torch")
    frames = torch.arange(SAMPLE_RATE, dtype=torch.float32)
    audio = torch.sin(2.0 * math.pi * 440.0 * frames / SAMPLE_RATE) * 0.25

    result = cast(NDArray[np.float32], rubband.stretch(audio, SAMPLE_RATE).numpy())

    assert result.shape == (SAMPLE_RATE,)
    assert np.max(np.abs(result)) > 0.1


def test_native_rejects_non_contiguous_torch_cpu_tensor() -> None:
    torch = pytest.importorskip("torch")
    audio = torch.zeros((2, SAMPLE_RATE), dtype=torch.float32).transpose(0, 1)

    with pytest.raises(TypeError):
        rubband.stretch(audio, SAMPLE_RATE)


def test_native_rejects_torch_cuda_tensor() -> None:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    audio = torch.zeros(SAMPLE_RATE, dtype=torch.float32, device="cuda")

    with pytest.raises(TypeError):
        rubband.stretch(audio, SAMPLE_RATE)


def test_native_stretch_returns_float32_audio_buffer() -> None:
    result = rubband.stretch(sine_wave(seconds=1.0), SAMPLE_RATE)
    view = memoryview(result.memoryview())  # ty: ignore[invalid-argument-type]

    assert isinstance(result, rubband.AudioBuffer)
    assert result.dtype == "float32"
    assert result.frames == SAMPLE_RATE
    assert result.channels == 1
    assert result.shape == (SAMPLE_RATE,)
    assert view.format == "f"
    assert view.itemsize == 4
    assert view.ndim == 1
    assert view.c_contiguous


def test_native_audio_buffer_converts_to_torch_tensor() -> None:
    torch = pytest.importorskip("torch")
    result = rubband.stretch(sine_wave(seconds=1.0), SAMPLE_RATE)

    tensor = result.torch()

    assert tuple(tensor.shape) == (SAMPLE_RATE,)  # ty: ignore[unresolved-attribute]
    assert tensor.dtype == torch.float32  # ty: ignore[unresolved-attribute]


def test_native_stretcher_regression(file_regression: FileRegressionFixture) -> None:
    audio = sine_wave(seconds=1.0)
    stretcher = rubband.Stretcher(SAMPLE_RATE, 1)

    stretcher.study(audio, final=True)
    stretcher.process(audio, final=True)
    result = cast(NDArray[np.float32], stretcher.retrieve().numpy())

    assert result.shape == (SAMPLE_RATE, 1)
    assert np.max(np.abs(result)) > 0.1
    file_regression.check(
        wav_bytes(result[:, 0]),
        extension=".wav",
        binary=True,
    )


def test_native_stretcher_retrieve_returns_rank_two_audio_buffer() -> None:
    audio = sine_wave(seconds=1.0)
    stretcher = rubband.Stretcher(SAMPLE_RATE, 1)

    stretcher.study(audio, final=True)
    stretcher.process(audio, final=True)
    result = stretcher.retrieve()
    view = memoryview(result.memoryview())  # ty: ignore[invalid-argument-type]

    assert isinstance(result, rubband.AudioBuffer)
    assert result.dtype == "float32"
    assert result.frames == SAMPLE_RATE
    assert result.channels == 1
    assert result.shape == (SAMPLE_RATE, 1)
    assert view.format == "f"
    assert view.itemsize == 4
    assert view.ndim == 2
    assert view.c_contiguous


def test_native_stretcher_accepts_dynamic_ratio_setters() -> None:
    stretcher = rubband.Stretcher(
        SAMPLE_RATE,
        1,
        options=rubband.Options(
            process=rubband.ProcessOption.real_time,
        ),
    )

    stretcher.set_time_ratio(0.75)
    stretcher.set_pitch_scale(1.5)
    assert stretcher.get_time_ratio() == 0.75
    assert stretcher.get_pitch_scale() == 1.5


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
        cast(
            NDArray[np.float32],
            rubband.stretch(
                audio,
                SAMPLE_RATE,
                time_ratio=1.25,
                pitch_scale=2.0,
            ).numpy(),
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

    result = cast(
        NDArray[np.float32],
        rubband.stretch(
            audio,
            sample_rate,
            pitch_scale=2.0 ** (semitones / 12.0),
        ).numpy(),
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


def dominant_frequency(
    audio: NDArray[np.float32],
    sample_rate: int = SAMPLE_RATE,
) -> float:
    windowed = audio * np.hanning(audio.shape[0])
    spectrum = np.fft.rfft(windowed)
    frequencies = np.fft.rfftfreq(audio.shape[0], 1.0 / sample_rate)
    return float(frequencies[int(np.argmax(np.abs(spectrum)))])


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


class DLPackAudio:
    def __init__(self, audio: NDArray[np.float32]) -> None:
        self.audio = audio

    def __dlpack__(
        self,
        stream: object = None,
        max_version: tuple[int, int] | None = None,
    ) -> object:
        return self.audio.__dlpack__(stream=stream, max_version=max_version)

    def __dlpack_device__(self) -> tuple[int, int]:
        return self.audio.__dlpack_device__()
