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
from rubband import _native

SAMPLE_RATE = 48_000


def test_stretch_accepts_one_second_mono_audio(
    monkeypatch: pytest.MonkeyPatch,
    file_regression: FileRegressionFixture,
) -> None:
    audio = sine_wave(seconds=1.0)

    def stretch_float32(
        backend_audio: NDArray[np.float32],
        sample_rate: int,
        time_ratio: float,
        pitch_scale: float,
        option_flags: int,
    ) -> NDArray[np.float32]:
        assert backend_audio.shape == (SAMPLE_RATE,)
        assert sample_rate == SAMPLE_RATE
        assert time_ratio == 1.25
        assert pitch_scale == 0.5
        assert option_flags == rubband.Options().option_flags
        return backend_audio.copy()

    monkeypatch.setattr(_native, "stretch_float32", stretch_float32)

    result = rubband.stretch(
        audio,
        SAMPLE_RATE,
        time_ratio=1.25,
        pitch_scale=0.5,
    )
    audio_result = cast(NDArray[np.float32], result.numpy())

    assert result.shape == (SAMPLE_RATE,)
    file_regression.check(
        wav_bytes(audio_result),
        extension=".wav",
        binary=True,
    )


def test_stretch_accepts_one_second_stereo_audio(
    monkeypatch: pytest.MonkeyPatch,
    file_regression: FileRegressionFixture,
) -> None:
    audio = np.column_stack((sine_wave(seconds=1.0), sine_wave(seconds=1.0, hz=660)))

    def stretch_float32(
        backend_audio: NDArray[np.float32],
        sample_rate: int,
        time_ratio: float,
        pitch_scale: float,
        option_flags: int,
    ) -> NDArray[np.float32]:
        assert sample_rate == SAMPLE_RATE
        assert time_ratio == 1.0
        assert pitch_scale == 2.0
        assert option_flags == rubband.Options().option_flags
        return backend_audio.copy()

    monkeypatch.setattr(_native, "stretch_float32", stretch_float32)

    result = rubband.stretch(
        np.ascontiguousarray(audio, dtype=np.float32),
        SAMPLE_RATE,
        pitch_scale=2.0,
    )
    audio_result = cast(NDArray[np.float32], result.numpy())

    assert result.shape == (SAMPLE_RATE, 2)
    file_regression.check(
        wav_bytes(audio_result),
        extension=".wav",
        binary=True,
    )


def test_windows_dll_directories_include_vcpkg_root(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("VCPKG_INSTALLATION_ROOT", str(tmp_path))

    directories = _native._windows_dll_directories()

    assert Path(_native.__file__).resolve().parent in directories
    assert tmp_path / "installed" / "x64-windows" / "bin" in directories


def test_stretch_accepts_array_buffer_audio(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    audio = array("f", [0.0] * SAMPLE_RATE)

    def stretch_float32(
        backend_audio: object,
        sample_rate: int,
        time_ratio: float,
        pitch_scale: float,
        option_flags: int,
    ) -> array[float]:
        view = memoryview(backend_audio)  # ty: ignore[invalid-argument-type]
        assert view.format == "f"
        assert view.shape == (SAMPLE_RATE,)
        return array("f", [0.0] * SAMPLE_RATE)

    monkeypatch.setattr(_native, "stretch_float32", stretch_float32)

    result = rubband.stretch(audio, SAMPLE_RATE)

    view = memoryview(result.memoryview())  # ty: ignore[invalid-argument-type]
    assert view.format == "f"
    assert view.shape == (SAMPLE_RATE,)


def test_stretch_accepts_memoryview_audio(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    audio = memoryview(array("f", [0.0] * SAMPLE_RATE))

    def stretch_float32(
        backend_audio: object,
        sample_rate: int,
        time_ratio: float,
        pitch_scale: float,
        option_flags: int,
    ) -> array[float]:
        assert backend_audio is audio
        return array("f", [0.0] * SAMPLE_RATE)

    monkeypatch.setattr(_native, "stretch_float32", stretch_float32)

    result = rubband.stretch(audio, SAMPLE_RATE)

    view = memoryview(result.memoryview())  # ty: ignore[invalid-argument-type]
    assert view.shape == (SAMPLE_RATE,)


def test_stretch_rejects_non_array_protocol_audio() -> None:
    with pytest.raises(TypeError, match="DLPack or the Python buffer protocol"):
        rubband.stretch(
            [0.0],  # type: ignore[arg-type]
            SAMPLE_RATE,
        )


def test_stretch_rejects_non_float32_buffer_audio() -> None:
    with pytest.raises(TypeError, match="float32"):
        rubband.stretch(array("d", [0.0] * SAMPLE_RATE), SAMPLE_RATE)


def test_stretch_rejects_non_float32_audio() -> None:
    audio = np.zeros(SAMPLE_RATE, dtype=np.float64)

    with pytest.raises(TypeError, match="float32"):
        rubband.stretch(
            audio,  # type: ignore[arg-type]
            SAMPLE_RATE,
        )


def test_stretch_rejects_non_contiguous_audio() -> None:
    audio = np.zeros((SAMPLE_RATE, 2), dtype=np.float32)[::2]

    with pytest.raises(ValueError, match="C-contiguous"):
        rubband.stretch(audio, SAMPLE_RATE)


def test_stretch_rejects_empty_audio() -> None:
    with pytest.raises(ValueError, match="at least one frame"):
        rubband.stretch(np.zeros(0, dtype=np.float32), SAMPLE_RATE)


def test_stretch_rejects_empty_channel_dimension() -> None:
    audio = np.zeros((SAMPLE_RATE, 0), dtype=np.float32)

    with pytest.raises(ValueError, match="at least one channel"):
        rubband.stretch(audio, SAMPLE_RATE)


def test_stretch_rejects_rank_three_audio() -> None:
    audio = np.zeros((SAMPLE_RATE, 1, 1), dtype=np.float32)

    with pytest.raises(ValueError, match=r"shape \(frames,\) or \(frames, channels\)"):
        rubband.stretch(audio, SAMPLE_RATE)


def test_stretch_rejects_out_of_range_sample_rate() -> None:
    with pytest.raises(ValueError, match="between 8000 and 192000"):
        rubband.Stretcher(7999, 1)


def test_stretch_rejects_non_positive_ratios() -> None:
    with pytest.raises(ValueError, match="ratio"):
        rubband.stretch(
            np.zeros(SAMPLE_RATE, dtype=np.float32),
            SAMPLE_RATE,
            time_ratio=0,
        )
    with pytest.raises(ValueError, match="ratio"):
        rubband.Stretcher(SAMPLE_RATE, 1, initial_pitch_scale=0)


def test_stretch_options_represent_all_rubber_band_option_groups() -> None:
    options = rubband.Options(
        process=rubband.ProcessOption.real_time,
        stretch=rubband.StretchOption.precise,
        transients=rubband.TransientsOption.smooth,
        detector=rubband.DetectorOption.soft,
        phase=rubband.PhaseOption.independent,
        threading=rubband.ThreadingOption.always,
        window=rubband.WindowOption.long,
        smoothing=rubband.SmoothingOption.on,
        formant=rubband.FormantOption.preserved,
        pitch=rubband.PitchOption.high_consistency,
        channels=rubband.ChannelsOption.together,
        engine=rubband.EngineOption.finer,
    )

    assert options.option_flags == 0x35A22A11


def test_stretch_options_include_rubber_band_presets() -> None:
    options = rubband.Options(
        preset=rubband.PresetOption.percussive,
    )

    assert options.option_flags == 0x32112000


def test_live_options_represent_rubber_band_live_option_groups() -> None:
    options = rubband.LiveOptions(
        window=rubband.LiveWindowOption.medium,
        formant=rubband.LiveFormantOption.preserved,
        channels=rubband.LiveChannelsOption.together,
    )

    assert options.option_flags == 0x11100000


def test_metadata_returns_accessors(monkeypatch: pytest.MonkeyPatch) -> None:
    options = rubband.Options()

    def metadata(
        sample_rate: int,
        channels: int,
        time_ratio: float,
        pitch_scale: float,
        option_flags: int,
    ) -> dict[str, int | float]:
        assert sample_rate == SAMPLE_RATE
        assert channels == 2
        assert time_ratio == 1.25
        assert pitch_scale == 2.0
        assert option_flags == options.option_flags
        return {
            "engine_version": 2,
            "available": 0,
            "preferred_start_pad": 128,
            "start_delay": 64,
            "time_ratio": 1.25,
            "pitch_scale": 2.0,
        }

    monkeypatch.setattr(_native, "metadata", metadata)

    result = rubband.metadata(
        SAMPLE_RATE,
        channels=2,
        time_ratio=1.25,
        pitch_scale=2.0,
        options=options,
    )

    assert result == rubband.RubberBandMetadata(
        engine_version=2,
        available=0,
        preferred_start_pad=128,
        start_delay=64,
        time_ratio=1.25,
        pitch_scale=2.0,
    )


def test_stretcher_mirrors_streaming_lifecycle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    options = rubband.Options()

    class NativeStretcher:
        def __init__(
            self,
            sample_rate: int,
            channels: int,
            time_ratio: float,
            pitch_scale: float,
            option_flags: int,
        ) -> None:
            assert sample_rate == SAMPLE_RATE
            assert channels == 2
            assert time_ratio == 1.25
            assert pitch_scale == 2.0
            assert option_flags == options.option_flags
            self.studied = False
            self.processed = False
            self.time_ratio = time_ratio
            self.pitch_scale = pitch_scale

        def study(self, audio: NDArray[np.float32], final: bool) -> None:
            assert audio.shape == (SAMPLE_RATE, 2)
            assert not final
            self.studied = True

        def process(self, audio: NDArray[np.float32], final: bool) -> None:
            assert audio.shape == (SAMPLE_RATE, 2)
            assert final
            self.processed = True

        def set_time_ratio(self, ratio: float) -> None:
            self.time_ratio = ratio

        def set_pitch_scale(self, scale: float) -> None:
            self.pitch_scale = scale

        def reset(self) -> None:
            self.studied = False
            self.processed = False

        def set_formant_scale(self, scale: float) -> None:
            self.formant_scale = scale

        def get_time_ratio(self) -> float:
            return self.time_ratio

        def get_pitch_scale(self) -> float:
            return self.pitch_scale

        def get_formant_scale(self) -> float:
            return getattr(self, "formant_scale", 0.0)

        def get_preferred_start_pad(self) -> int:
            return 128

        def get_start_delay(self) -> int:
            return 64

        def get_latency(self) -> int:
            return 64

        def get_channel_count(self) -> int:
            return 2

        def set_expected_input_duration(self, samples: int) -> None:
            self.expected_input_duration = samples

        def set_max_process_size(self, samples: int) -> None:
            self.max_process_size = samples

        def get_process_size_limit(self) -> int:
            return 524_288

        def get_samples_required(self) -> int:
            return 256

        def available(self) -> int:
            return 3

        def retrieve(self) -> NDArray[np.float32]:
            assert self.studied
            assert self.processed
            return np.zeros((3, 2), dtype=np.float32)

    monkeypatch.setattr(_native, "Stretcher", NativeStretcher)
    audio = np.ascontiguousarray(
        np.column_stack((sine_wave(seconds=1.0), sine_wave(seconds=1.0, hz=660))),
        dtype=np.float32,
    )

    stretcher = rubband.Stretcher(
        SAMPLE_RATE,
        2,
        options=options,
        initial_time_ratio=1.25,
        initial_pitch_scale=2.0,
    )
    stretcher.study(audio)
    stretcher.process(audio, final=True)

    assert stretcher.available() == 3
    result = stretcher.retrieve()
    assert result.shape == (3, 2)


def test_stretcher_exposes_original_accessor_methods(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(_native, "Stretcher", FakeNativeStretcher)
    stretcher = rubband.Stretcher(
        SAMPLE_RATE,
        2,
        initial_time_ratio=1.25,
        initial_pitch_scale=2.0,
    )

    assert stretcher.get_time_ratio() == 1.25
    assert stretcher.get_pitch_scale() == 2.0
    assert stretcher.get_formant_scale() == 0.0
    assert stretcher.get_preferred_start_pad() == 128
    assert stretcher.get_start_delay() == 64
    assert stretcher.get_latency() == 64
    assert stretcher.get_channel_count() == 2
    assert stretcher.get_process_size_limit() == 524_288
    assert stretcher.get_samples_required() == 256

    stretcher.set_formant_scale(0.5)
    stretcher.set_expected_input_duration(48_000)
    stretcher.set_max_process_size(1_024)
    stretcher.set_phase_option(rubband.PhaseOption.independent)
    stretcher.set_formant_option(rubband.FormantOption.preserved)
    native = cast(FakeNativeStretcher, stretcher.native)
    assert native.formant_scale == 0.5
    assert native.expected_input_duration == 48_000
    assert native.max_process_size == 1_024
    assert native.phase_option == 0x00002000
    assert native.formant_option == 0x01000000


def test_stretcher_exposes_rubber_band_4_methods(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(_native, "Stretcher", FakeNativeStretcher)
    stretcher = rubband.Stretcher(SAMPLE_RATE, 1)

    stretcher.set_key_frame_map({0: 0, 48_000: 72_000})
    stretcher.set_frequency_cutoff(1, 880.0)
    stretcher.set_debug_level(1)
    rubband.Stretcher.set_default_debug_level(0)

    native = cast(FakeNativeStretcher, stretcher.native)
    assert stretcher.get_engine_version() == 3
    assert stretcher.get_frequency_cutoff(1) == 880.0
    assert stretcher.get_input_increment() == 512
    assert stretcher.get_output_increments() == [128, 256]
    assert stretcher.get_phase_reset_curve() == [0.0, 0.5]
    assert stretcher.get_exact_time_points() == [100, 200]
    assert native.key_frame_map == {0: 0, 48_000: 72_000}
    assert native.debug_level == 1


def test_stretcher_rejects_invalid_key_frame_map(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(_native, "Stretcher", FakeNativeStretcher)
    stretcher = rubband.Stretcher(SAMPLE_RATE, 1)

    with pytest.raises(ValueError, match="sample count"):
        stretcher.set_key_frame_map({0: -1})

    stretcher.process(sine_wave(seconds=1.0), final=False)
    with pytest.raises(ValueError, match="key frame map"):
        stretcher.set_key_frame_map({0: 0})


def test_real_time_stretcher_accepts_original_option_mutators(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(_native, "Stretcher", FakeNativeStretcher)
    stretcher = rubband.Stretcher(
        SAMPLE_RATE,
        1,
        options=rubband.Options(
            process=rubband.ProcessOption.real_time,
        ),
    )

    stretcher.set_transients_option(rubband.TransientsOption.smooth)
    stretcher.set_detector_option(rubband.DetectorOption.soft)
    stretcher.set_pitch_option(rubband.PitchOption.high_consistency)

    native = cast(FakeNativeStretcher, stretcher.native)
    assert native.transients_option == 0x00000200
    assert native.detector_option == 0x00000800
    assert native.pitch_option == 0x04000000


def test_offline_stretcher_rejects_real_time_only_option_mutators(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(_native, "Stretcher", FakeNativeStretcher)
    stretcher = rubband.Stretcher(SAMPLE_RATE, 1)

    with pytest.raises(ValueError, match="offline"):
        stretcher.set_transients_option(rubband.TransientsOption.smooth)
    with pytest.raises(ValueError, match="offline"):
        stretcher.set_detector_option(rubband.DetectorOption.soft)
    with pytest.raises(ValueError, match="offline"):
        stretcher.set_pitch_option(rubband.PitchOption.high_consistency)


def test_real_time_stretcher_accepts_dynamic_ratio_changes_after_processing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(_native, "Stretcher", FakeNativeStretcher)
    stretcher = rubband.Stretcher(
        SAMPLE_RATE,
        1,
        options=rubband.Options(
            process=rubband.ProcessOption.real_time,
        ),
    )

    stretcher.process(sine_wave(seconds=1.0), final=False)
    stretcher.set_time_ratio(0.75)
    stretcher.set_pitch_scale(1.5)

    native = cast(FakeNativeStretcher, stretcher.native)
    assert native.time_ratio == 0.75
    assert native.pitch_scale == 1.5


def test_offline_stretcher_rejects_dynamic_ratio_changes_after_processing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(_native, "Stretcher", FakeNativeStretcher)
    stretcher = rubband.Stretcher(SAMPLE_RATE, 1)

    stretcher.process(sine_wave(seconds=1.0), final=False)

    with pytest.raises(ValueError, match="offline"):
        stretcher.set_time_ratio(0.75)
    with pytest.raises(ValueError, match="offline"):
        stretcher.set_pitch_scale(1.5)


def test_stretcher_rejects_non_finite_dynamic_ratios() -> None:
    stretcher = rubband.Stretcher(SAMPLE_RATE, 1)

    with pytest.raises(ValueError, match="finite"):
        stretcher.set_time_ratio(math.inf)
    with pytest.raises(ValueError, match="finite"):
        stretcher.set_pitch_scale(math.nan)


def test_stretcher_rejects_channel_mismatch() -> None:
    stretcher = rubband.Stretcher(SAMPLE_RATE, 2)

    with pytest.raises(ValueError, match="channel count"):
        stretcher.study(sine_wave(seconds=1.0))


def test_stretcher_rejects_invalid_sample_counts() -> None:
    stretcher = rubband.Stretcher(SAMPLE_RATE, 1)

    with pytest.raises(ValueError, match="sample count"):
        stretcher.set_expected_input_duration(-1)
    with pytest.raises(ValueError, match="sample count"):
        stretcher.set_max_process_size(-1)


def test_live_shifter_mirrors_rubber_band_live_lifecycle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(_native, "LiveShifter", FakeNativeLiveShifter)
    audio = sine_wave(seconds=1.0)[:128]
    output = np.zeros(128, dtype=np.float32)
    shifter = rubband.LiveShifter(
        SAMPLE_RATE,
        1,
        options=rubband.LiveOptions(
            window=rubband.LiveWindowOption.medium,
            formant=rubband.LiveFormantOption.preserved,
            channels=rubband.LiveChannelsOption.together,
        ),
    )

    shifter.set_pitch_scale(1.5)
    shifter.set_formant_scale(0.75)
    shifter.set_formant_option(rubband.LiveFormantOption.shifted)
    shifter.set_debug_level(1)
    rubband.LiveShifter.set_default_debug_level(0)
    result = shifter.shift(audio)
    shifter.shift_into(audio, output)

    native = cast(FakeNativeLiveShifter, shifter.native)
    assert shifter.get_pitch_scale() == 1.5
    assert shifter.get_formant_scale() == 0.75
    assert shifter.get_start_delay() == 32
    assert shifter.get_channel_count() == 1
    assert shifter.get_block_size() == 128
    assert result.shape == (128,)
    np.testing.assert_array_equal(output, audio)
    assert native.option_flags == 0x11100000
    assert native.formant_option == 0
    assert native.debug_level == 1


def test_live_shifter_rejects_wrong_block_size(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(_native, "LiveShifter", FakeNativeLiveShifter)
    shifter = rubband.LiveShifter(SAMPLE_RATE, 1)

    with pytest.raises(ValueError, match="block size"):
        shifter.shift(np.zeros(127, dtype=np.float32))


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


def test_documented_public_callables_have_docstrings() -> None:
    public_callables = [
        rubband.stretch,
        rubband.metadata,
        rubband.AudioBuffer.validate_data,
        rubband.AudioBuffer.dtype.fget,
        rubband.AudioBuffer.frames.fget,
        rubband.AudioBuffer.channels.fget,
        rubband.AudioBuffer.shape.fget,
        rubband.AudioBuffer.memoryview,
        rubband.AudioBuffer.numpy,
        rubband.AudioBuffer.torch,
        rubband.Options.option_flags.fget,
        rubband.LiveOptions.option_flags.fget,
        rubband.Stretcher.study,
        rubband.Stretcher.process,
        rubband.Stretcher.reset,
        rubband.Stretcher.set_time_ratio,
        rubband.Stretcher.set_pitch_scale,
        rubband.Stretcher.set_formant_scale,
        rubband.Stretcher.set_transients_option,
        rubband.Stretcher.set_detector_option,
        rubband.Stretcher.set_phase_option,
        rubband.Stretcher.set_formant_option,
        rubband.Stretcher.set_pitch_option,
        rubband.Stretcher.get_time_ratio,
        rubband.Stretcher.get_pitch_scale,
        rubband.Stretcher.get_formant_scale,
        rubband.Stretcher.get_preferred_start_pad,
        rubband.Stretcher.get_start_delay,
        rubband.Stretcher.get_latency,
        rubband.Stretcher.get_channel_count,
        rubband.Stretcher.set_expected_input_duration,
        rubband.Stretcher.set_max_process_size,
        rubband.Stretcher.get_process_size_limit,
        rubband.Stretcher.get_samples_required,
        rubband.Stretcher.available,
        rubband.Stretcher.retrieve,
        rubband.Stretcher.get_engine_version,
        rubband.Stretcher.set_key_frame_map,
        rubband.Stretcher.get_frequency_cutoff,
        rubband.Stretcher.set_frequency_cutoff,
        rubband.Stretcher.get_input_increment,
        rubband.Stretcher.get_output_increments,
        rubband.Stretcher.get_phase_reset_curve,
        rubband.Stretcher.get_exact_time_points,
        rubband.Stretcher.set_debug_level,
        rubband.Stretcher.set_default_debug_level,
        rubband.LiveShifter.reset,
        rubband.LiveShifter.set_pitch_scale,
        rubband.LiveShifter.set_formant_scale,
        rubband.LiveShifter.get_pitch_scale,
        rubband.LiveShifter.get_formant_scale,
        rubband.LiveShifter.get_start_delay,
        rubband.LiveShifter.get_channel_count,
        rubband.LiveShifter.set_formant_option,
        rubband.LiveShifter.get_block_size,
        rubband.LiveShifter.shift,
        rubband.LiveShifter.shift_into,
        rubband.LiveShifter.set_debug_level,
        rubband.LiveShifter.set_default_debug_level,
    ]

    for function in public_callables:
        assert function is not None
        assert function.__doc__


class FakeNativeStretcher:
    def __init__(
        self,
        sample_rate: int,
        channels: int,
        time_ratio: float,
        pitch_scale: float,
        option_flags: int,
    ) -> None:
        self.sample_rate = sample_rate
        self.channels = channels
        self.time_ratio = time_ratio
        self.pitch_scale = pitch_scale
        self.option_flags = option_flags
        self.formant_scale = 0.0
        self.transients_option = 0
        self.detector_option = 0
        self.phase_option = 0
        self.formant_option = 0
        self.pitch_option = 0
        self.expected_input_duration = 0
        self.max_process_size = 0
        self.started = False
        self.key_frame_map: dict[int, int] = {}
        self.frequency_cutoffs: dict[int, float] = {}
        self.debug_level = 0

    def study(self, audio: NDArray[np.float32], final: bool) -> None:
        self.started = True

    def process(self, audio: NDArray[np.float32], final: bool) -> None:
        self.started = True

    def reset(self) -> None:
        self.started = False

    def set_time_ratio(self, ratio: float) -> None:
        self.time_ratio = ratio

    def set_pitch_scale(self, scale: float) -> None:
        self.pitch_scale = scale

    def set_formant_scale(self, scale: float) -> None:
        self.formant_scale = scale

    def set_transients_option(self, options: int) -> None:
        self.transients_option = options

    def set_detector_option(self, options: int) -> None:
        self.detector_option = options

    def set_phase_option(self, options: int) -> None:
        self.phase_option = options

    def set_formant_option(self, options: int) -> None:
        self.formant_option = options

    def set_pitch_option(self, options: int) -> None:
        self.pitch_option = options

    def get_time_ratio(self) -> float:
        return self.time_ratio

    def get_pitch_scale(self) -> float:
        return self.pitch_scale

    def get_formant_scale(self) -> float:
        return self.formant_scale

    def get_preferred_start_pad(self) -> int:
        return 128

    def get_start_delay(self) -> int:
        return 64

    def get_latency(self) -> int:
        return 64

    def get_channel_count(self) -> int:
        return self.channels

    def set_expected_input_duration(self, samples: int) -> None:
        self.expected_input_duration = samples

    def set_max_process_size(self, samples: int) -> None:
        self.max_process_size = samples

    def get_process_size_limit(self) -> int:
        return 524_288

    def get_samples_required(self) -> int:
        return 256

    def available(self) -> int:
        return 0

    def retrieve(self) -> NDArray[np.float32]:
        return np.zeros((0, self.channels), dtype=np.float32)

    def get_engine_version(self) -> int:
        return 3

    def set_key_frame_map(self, key_frames: dict[int, int]) -> None:
        self.key_frame_map = key_frames

    def get_frequency_cutoff(self, n: int) -> float:
        return self.frequency_cutoffs[n]

    def set_frequency_cutoff(self, n: int, frequency: float) -> None:
        self.frequency_cutoffs[n] = frequency

    def get_input_increment(self) -> int:
        return 512

    def get_output_increments(self) -> list[int]:
        return [128, 256]

    def get_phase_reset_curve(self) -> list[float]:
        return [0.0, 0.5]

    def get_exact_time_points(self) -> list[int]:
        return [100, 200]

    def set_debug_level(self, level: int) -> None:
        self.debug_level = level

    @staticmethod
    def set_default_debug_level(level: int) -> None:
        return None


class FakeNativeLiveShifter:
    def __init__(self, sample_rate: int, channels: int, option_flags: int) -> None:
        self.sample_rate = sample_rate
        self.channels = channels
        self.option_flags = option_flags
        self.pitch_scale = 1.0
        self.formant_scale = 0.0
        self.formant_option = 0
        self.debug_level = 0

    def reset(self) -> None:
        return None

    def set_pitch_scale(self, scale: float) -> None:
        self.pitch_scale = scale

    def set_formant_scale(self, scale: float) -> None:
        self.formant_scale = scale

    def get_pitch_scale(self) -> float:
        return self.pitch_scale

    def get_formant_scale(self) -> float:
        return self.formant_scale

    def get_start_delay(self) -> int:
        return 32

    def get_channel_count(self) -> int:
        return self.channels

    def set_formant_option(self, options: int) -> None:
        self.formant_option = options

    def get_block_size(self) -> int:
        return 128

    def shift(self, audio: NDArray[np.float32]) -> NDArray[np.float32]:
        return audio.copy()

    def shift_into(
        self,
        audio: NDArray[np.float32],
        output: NDArray[np.float32],
    ) -> None:
        output[:] = audio

    def set_debug_level(self, level: int) -> None:
        self.debug_level = level

    @staticmethod
    def set_default_debug_level(level: int) -> None:
        return None


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
