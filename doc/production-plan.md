# Rubband Production Plan

## Goal

Turn the current Rubber Band/nanobind demo into a production-quality Python
library for Windows, Linux, and macOS.

The library should stay close to Rubber Band's original API. It should add only
the Python wrapping needed for safe array handling, packaging, and testing.

## Non-Goals

- GPU processing.
- PyTorch autograd.
- Streaming through Python async APIs.
- A high-level effects framework.
- New DSP behavior not provided by Rubber Band.

## Public API Shape

Keep the basic practical API:

```python
rubband.stretch(
    audio,
    sample_rate,
    time_ratio=1.0,
    pitch_scale=1.0,
    options=rubband.Options(),
)
```

Inputs:

- 1D mono audio: `(frames,)`
- 2D audio: `(frames, channels)`
- `float32` first
- contiguous CPU memory only
- array-protocol compatible input where possible

Outputs:

- Return an array in the same frame/channel layout.
- Initially return NumPy arrays for all inputs.
- Later preserve the input framework only when it can be done safely without
  pretending to support GPU memory.

The public API should not hide Rubber Band concepts. If Rubber Band exposes a
meaningful option, expose that option with the same name or a close Pythonic
equivalent.

## Array Protocol Plan

Use nanobind's ndarray support as the native boundary.

Supported input contract:

- CPU memory.
- Contiguous layout.
- `float32`.
- Compatible with NumPy and any framework that can expose CPU array memory
  through the protocols nanobind supports.

Validation should reject:

- GPU tensors.
- Non-contiguous arrays.
- Unsupported dtypes.
- Object arrays.
- Unknown or non-readable memory layouts.

Do not silently copy framework arrays in the native layer. Copies should happen
only in explicit Python helper functions or clearly documented convenience
wrappers. The core binding should represent exactly what Rubber Band can consume:
contiguous CPU sample buffers.

## Rubber Band Coverage

Map Rubber Band's main C++ API in stages.

### Stage 1: Offline Stretching

Finish the current offline API:

- time ratio
- pitch scale
- mono and multichannel audio
- deterministic draining of available output
- regression tests with generated and real audio

The native implementation should keep the conversion boundary simple:

- Python receives interleaved `(frames, channels)` arrays.
- C++ converts to Rubber Band's planar `float *const *` input.
- C++ returns interleaved `(frames, channels)` arrays.

### Stage 2: Options

Expose Rubber Band options without inventing a separate effects vocabulary.

Likely options to wrap:

- process mode: offline or real-time
- threading choice
- engine choice when available
- transient handling
- detector choice
- phase handling
- window length
- smoothing
- formant preservation
- pitch option behavior

Represent options as a small Python data object or enum-backed flags that convert
directly to Rubber Band's option bitmask. Keep names traceable to Rubber Band.

### Stage 3: Metadata And Ratios

Expose safe read-only accessors where useful:

- engine version if available
- available output count
- latency/start delay if relevant to streaming
- effective time ratio and pitch scale

Do not add stateful Python objects until the stateless offline function is solid.

### Stage 4: Stateful Streaming Wrapper

Add a thin class around `RubberBandStretcher` only when needed:

```python
stretcher = rubband.Stretcher(sample_rate, channels, options=...)
stretcher.study(audio, final=False)
stretcher.process(audio, final=False)
out = stretcher.retrieve()
```

This should mirror Rubber Band's lifecycle. Avoid an invented buffering model.

## Cross-Platform Packaging

Keep `scikit-build-core`, CMake, and nanobind.

### macOS

- Support arm64 and x86_64.
- Prefer Homebrew for local development.
- For wheels, decide whether to vendor Rubber Band or link dynamically.
- Use `delocate` if shipping wheels with bundled dynamic libraries.

### Linux

- Build manylinux wheels.
- Use `auditwheel` for bundled shared libraries.
- Decide whether to build Rubber Band from source in CI or use system packages
  only for source builds.

### Windows

- Build wheels with MSVC.
- Build or vendor Rubber Band and its dependencies in CI.
- Ensure the wheel includes required `.dll` files.
- Add import-time tests that verify the extension loads on Windows runners.

Packaging policy should be explicit:

- Source distributions may require system `librubberband`.
- Wheels should work without users installing Rubber Band separately, if license
  and build complexity permit.

## Build System Work

Immediate tasks:

- Add CMake checks that fail with a clear message when Rubber Band is missing.
- Keep `pkg-config` support for Unix development.
- Add Windows discovery for vendored or CMake-built Rubber Band.
- Generate or include type stubs for the native extension.
- Add CI matrix builds for macOS, Linux, and Windows.

Avoid adding optional build paths until one wheel path works end to end.

## Testing Strategy

Use three layers of tests.

### Python API Tests

- validation errors
- dtype and shape handling
- mono and stereo layout
- array-protocol input acceptance once implemented

These can keep using a fake backend where appropriate.

### Native Regression Tests

- generated one-second WAVs at 48 kHz
- real stereo WAV regression input
- pitch shifts up/down 5 and 12 semitones
- time stretching
- prefix comparison for cases where Rubber Band has small tail-length variation

All audio regression files should be WAV files with at least one second of audio.

### Packaging Tests

- build wheel
- install wheel into a clean environment
- import `rubband`
- run a tiny native stretch

Run packaging tests on Windows, Linux, and macOS.

## Documentation

Document:

- accepted array shapes
- dtype requirements
- CPU-only behavior
- contiguous memory requirement
- how PyTorch CPU tensors can be passed once array-protocol support is added
- why GPU tensors are rejected
- how output length can vary slightly for some Rubber Band stereo operations
- how to install from source when Rubber Band is supplied by the OS/package manager

Do not document unsupported frameworks as working until tests prove them.

## Release Criteria

Before a production release:

- native tests pass on macOS, Linux, and Windows
- wheels install cleanly on all supported platforms
- source distribution builds with documented system dependencies
- README has installation instructions for each platform
- API reference covers every exposed Rubber Band option
- regression WAVs are stable enough for review
- no hidden copies in the core native path

## Additional work beyond the prompt

None.
