# rubband

Python bindings for Rubber Band audio pitch shifting and time stretching.

Documentation: https://rec.github.io/rubband/

The API accepts contiguous CPU `float32` audio through DLPack or the Python
buffer protocol:

```python
from array import array

import rubband

audio = array("f", [0.0] * 48_000)

shifted = rubband.stretch(
    audio,
    48_000,
    time_ratio=1.25,
    pitch_scale=1.0,
)
```

For stateful processing, use `Stretcher`:

```python
stretcher = rubband.Stretcher(
    48_000,
    2,
    options=rubband.Options(
        process=rubband.ProcessOption.real_time,
    ),
    initial_time_ratio=1.0,
    initial_pitch_scale=1.0,
)
stretcher.process(audio, final=False)
stretcher.set_pitch_scale(2.0)
stretcher.set_time_ratio(0.75)
```

Dynamic `set_pitch_scale()` and `set_time_ratio()` changes are intended for
real-time processing. In offline mode, call them before `study()` or
`process()`; after either call, rubband raises `ValueError`.

`Stretcher` also exposes original Rubber Band lifecycle and query methods such
as `reset()`, `get_time_ratio()`, `get_pitch_scale()`,
`get_preferred_start_pad()`, `get_start_delay()`, `get_channel_count()`,
`set_expected_input_duration()`, `set_max_process_size()`,
`get_samples_required()`, `set_transients_option()`, `set_detector_option()`,
`set_phase_option()`, `set_formant_option()`, and `set_pitch_option()`.

Input constraints:

- DLPack or Python buffer protocol input
- `float32` only
- CPU memory only
- shape `(frames,)` for mono or `(frames, channels)` for multichannel audio
- C-contiguous arrays only
- sample rates from 8,000 to 192,000 Hz

`stretch()` and `Stretcher.retrieve()` return `memoryview` objects over
C-contiguous `float32` output. Callers can convert those results into their
preferred array library.

The native backend is a nanobind extension over Rubber Band. Building it requires
`librubberband` to be installed and discoverable through `pkg-config`.

## Building

Local source builds need Rubber Band development headers and libraries:

- macOS: `brew install rubberband pkg-config`
- Linux: install your distribution's `librubberband-dev` or equivalent package
- Windows: install Rubber Band through `vcpkg` and pass its CMake toolchain file

Build artifacts with:

```sh
uv build --sdist --wheel --out-dir dist
```

Then smoke-test the wheel in a clean virtual environment:

```sh
uv run python scripts/smoke_wheel.py dist/rubband-*.whl
```

Release builds are tag-driven. Pushing a `v*` tag runs the GitHub Actions
workflow, builds platform artifacts, smoke-tests the wheel, writes checksums, and
publishes a GitHub release.
