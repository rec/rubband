# rubband

Python bindings for Rubber Band audio pitch shifting and time stretching.

The current API is NumPy-only:

```python
import rubband

shifted = rubband.stretch(
    audio,
    rubband.StretchOptions(
        sample_rate=48_000,
        time_ratio=1.25,
        pitch_scale=1.0,
    ),
)
```

Input constraints:

- `numpy.ndarray` only
- `float32` only
- CPU memory only
- shape `(frames,)` for mono or `(frames, channels)` for multichannel audio
- C-contiguous arrays only
- sample rates from 8,000 to 192,000 Hz

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
