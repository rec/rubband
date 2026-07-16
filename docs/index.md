# rubband

`rubband` is a Python binding for the Rubber Band audio time stretching and
pitch shifting library. Audio inputs can come from DLPack producers such as
NumPy and PyTorch, or from Python buffer protocol objects such as `array.array`.

The API stays close to Rubber Band's own model:

- `Options` contains Rubber Band option flags.
- `Stretcher` wraps the stateful Rubber Band stretcher.
- `LiveShifter` wraps Rubber Band 4.x's block-based live pitch shifter.
- `stretch()` provides a small offline convenience function.

See the [API reference](api.md) for generated documentation with signatures and
type annotations.

## Quick Example

```python
from array import array

import rubband

audio = array("f", [0.0] * 48_000)

shifted = rubband.stretch(
    audio,
    48_000,
    time_ratio=1.25,
    pitch_scale=2.0,
)

samples = shifted.memoryview()
```

## Input Constraints

- DLPack or Python buffer protocol input
- `float32` only
- CPU memory only
- shape `(frames,)` for mono or `(frames, channels)` for multichannel audio
- C-contiguous arrays only
- sample rates from 8,000 to 192,000 Hz

Outputs are `AudioBuffer` objects over C-contiguous `float32` audio. Use
`.memoryview()` for a zero-copy standard-library view, `.numpy()` for a NumPy
view, or `.torch()` for a PyTorch CPU tensor view.

## Stateful Processing

```python
stretcher = rubband.Stretcher(
    48_000,
    2,
    options=rubband.Options(process=rubband.ProcessOption.real_time),
)

stretcher.process(audio, final=False)
stretcher.set_pitch_scale(1.5)
result = stretcher.retrieve()
samples = result.memoryview()
```

Dynamic `set_pitch_scale()` and `set_time_ratio()` changes are intended for
real-time processing. In offline mode, call them before `study()` or
`process()`.

## Live Pitch Shifting

```python
shifter = rubband.LiveShifter(48_000, 1)
block_size = shifter.get_block_size()

block = audio[:block_size]
shifted = shifter.shift(block)
samples = shifted.memoryview()
```

`LiveShifter` follows Rubber Band 4.x's fixed-block live pitch-shifting API:
each `shift()` or `shift_into()` call must use exactly `get_block_size()`
frames. The native Rubber Band object is designed for live use, but Python
calls are not guaranteed hard real-time safe.
