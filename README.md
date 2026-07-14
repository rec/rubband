# rubband

Python bindings for Rubber Band audio pitch shifting and time stretching.

The current API is NumPy-only:

```python
import rubband

shifted = rubband.stretch(
    audio,
    sample_rate=48_000,
    time_ratio=1.25,
    pitch_scale=1.0,
)
```

Input constraints:

- `numpy.ndarray` only
- `float32` only
- CPU memory only
- shape `(frames,)` for mono or `(frames, channels)` for multichannel audio
- C-contiguous arrays only
- sample rates from 8,000 to 192,000 Hz

The native Rubber Band backend is not implemented yet. Calling `stretch` validates
the public API contract, then delegates to the native backend hook.
