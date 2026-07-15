# rubband

`rubband` is a NumPy-focused Python binding for the Rubber Band audio time
stretching and pitch shifting library.

The API stays close to Rubber Band's own model:

- `Options` contains Rubber Band option flags.
- `Stretcher` wraps the stateful Rubber Band stretcher.
- `stretch()` provides a small offline convenience function.

See the [API reference](api.md) for generated documentation with signatures and
type annotations.

## Quick Example

```python
import numpy as np
import rubband

audio = np.zeros(48_000, dtype=np.float32)

shifted = rubband.stretch(
    audio,
    48_000,
    time_ratio=1.25,
    pitch_scale=2.0,
)
```

## Input Constraints

- `numpy.ndarray` only
- `float32` only
- CPU memory only
- shape `(frames,)` for mono or `(frames, channels)` for multichannel audio
- C-contiguous arrays only
- sample rates from 8,000 to 192,000 Hz

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
```

Dynamic `set_pitch_scale()` and `set_time_ratio()` changes are intended for
real-time processing. In offline mode, call them before `study()` or
`process()`.
