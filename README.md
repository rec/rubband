# ➰ `rubband`: tensor-friendly bindings for the C++ Rubber Band pitch shifting library ➰

## What is this?

Rubber Band is a popular GPL C++ library for pitch shifting and time stretching with a
long history, but all the Python bindings for it are old and unsupported and only use
Rubber Band 3 or earlier - but version 4 has some significant new features

This new one uses [nanobind](https://nanobind.readthedocs.io/en/latest/) for the
bindings, and [DLPack](https://dmlc.github.io/dlpack/latest/) to interoperate with all
common tensor types.

## You wrote this rather fast. Is this AI slop?

I first started writing digital audio programs in the late 1970s, and I've written a
pretty huge number of programs of all types since then - [here's some recent
stuff.](https://github.com/rec) - and until June of 2026, every bit (hah!) was done by
hand.

`rubband` is my first entirely "vibe coded" library. While I carefully guided it at each
point, backed up and tried again a couple of times, and reviewed every line, the code
and all the documentation except this part was written by Codex.

I was positively surprised by the quality, and I think it is quite acceptable.  There
are copious tests, and I am very responsive to issues filed.

## If you're human, why is there an emoji in the description of this project?

I've been using emojis in project descriptions for about six years. They learned it from me.

#### Nothing below this line was written by a person.

------


## Public release notes

The first public release is intended to be a practical, minimal binding to the
Rubber Band stretcher. Binary wheels bundle the native Rubber Band 4.x library
used in release CI; source builds still link against the Rubber Band library
installed on the build machine.

### Installation

For a released binary wheel:

```sh
pip install rubband
```

When building from source, install Rubber Band 4.x first, then install
`rubband`.

On macOS:

```sh
brew install rubberband pkg-config
pip install rubband
```

On Linux, Rubber Band 4.0 or newer is required. Many stable distributions
still package older Rubber Band headers, so build and install Rubber Band 4.x
first if your distribution package is too old:

```sh
sudo apt-get update
sudo apt-get install -y build-essential curl meson ninja-build pkg-config
curl -L --fail --silent --show-error \
  https://github.com/breakfastquay/rubberband/archive/refs/tags/v4.0.0.tar.gz \
  | tar -xz
cd rubberband-4.0.0
meson setup build -Dauto_features=disabled -Ddefault_library=shared
ninja -C build
sudo ninja -C build install
sudo ldconfig
pip install rubband
```

On Windows, install Rubber Band with `vcpkg` before building from source:

```powershell
vcpkg install rubberband:x64-windows
$env:CMAKE_ARGS="-DCMAKE_TOOLCHAIN_FILE=$env:VCPKG_INSTALLATION_ROOT/scripts/buildsystems/vcpkg.cmake"
pip install rubband
```

### Input contract

Audio input must be:

- CPU memory
- contiguous
- `float32`
- mono shape `(frames,)` or multichannel shape `(frames, channels)`
- exposed through DLPack or the Python buffer protocol

This means contiguous NumPy arrays, PyTorch CPU tensors, `array.array("f")`,
and `memoryview` objects can work. CUDA tensors, non-contiguous views, and
non-`float32` arrays are rejected.

### Output contract

`stretch()` and `Stretcher.retrieve()` return an `AudioBuffer`.

```python
output = rubband.stretch(audio, 48_000, pitch_scale=2.0)

output.dtype       # "float32"
output.shape       # (frames,) or (frames, channels)
output.frames
output.channels
output.memoryview()
```

The conversion helpers are optional and import their array libraries lazily:

```python
numpy_audio = output.numpy()
torch_audio = output.torch()
```

NumPy and PyTorch are development/test dependencies, not runtime dependencies.

### Minimal examples

With the Python standard library:

```python
from array import array

import rubband

audio = array("f", [0.0] * 48_000)
output = rubband.stretch(audio, 48_000, time_ratio=1.25, pitch_scale=1.0)
samples = output.memoryview()
```

With NumPy:

```python
import numpy as np
import rubband

audio = np.zeros(48_000, dtype=np.float32)
output = rubband.stretch(audio, 48_000, pitch_scale=2.0)
numpy_audio = output.numpy()
```

With PyTorch CPU tensors:

```python
import torch
import rubband

audio = torch.zeros(48_000, dtype=torch.float32)
output = rubband.stretch(audio, 48_000, pitch_scale=0.5)
torch_audio = output.torch()
```

For stateful processing:

```python
import rubband

stretcher = rubband.Stretcher(
    48_000,
    1,
    options=rubband.Options(process=rubband.ProcessOption.real_time),
)

stretcher.process(audio, final=True)
output = stretcher.retrieve()
```

For Rubber Band 4.x live pitch shifting:

```python
import numpy as np
import rubband

shifter = rubband.LiveShifter(48_000, 1)
block_size = shifter.get_block_size()
audio = np.zeros(block_size, dtype=np.float32)

output = shifter.shift(audio)
```

`LiveShifter` uses Rubber Band's fixed-block live API. Each `shift()` or
`shift_into()` call must use exactly `get_block_size()` frames. The native
Rubber Band object is intended for live use, but Python calls are not
guaranteed hard real-time safe.

For diagnostics, `Stretcher` and `LiveShifter` accept an optional Python
logger callback:

```python
def logger(*values: object) -> None:
    print(values)

stretcher = rubband.Stretcher(48_000, 1, logger=logger)
stretcher.set_debug_level(1)
```

Rubber Band may call the logger from native processing code. `rubband`
acquires the Python GIL before invoking the callback, so this is useful for
debugging and diagnostics, but it is not hard real-time safe. Keep callbacks
short, avoid allocation-heavy work, and do not rely on them in a low-latency
audio callback. Exceptions raised by the logger propagate through the native
call that triggered the log message.

### Platform support

Release builds are configured for:

- macOS
- Linux
- Windows

Each platform wheel is built and smoke-tested in GitHub Actions against the
Rubber Band 4.x library installed or built by release CI. The release workflow
repairs those wheels so the required native shared libraries are bundled.
If a platform-specific release artifact is missing, that platform should be
treated as unsupported for that release.

### Documentation

Generated API documentation is published at:

https://rec.github.io/rubband/

### License

`rubband` is licensed under GPL-2.0-or-later. The repository contains the GPL
2.0 license text in `LICENSE`, and `pyproject.toml` uses the
`GPL-2.0-or-later` SPDX identifier.

Rubber Band is a separate native library with its own GPL-compatible license
terms. Binary wheels may include Rubber Band 4.x native binaries and related
shared-library dependencies; `rubband` includes a third-party notice file in
the package and distributes those binaries under their original license terms.
Source builds must have the Rubber Band 4.x library available at build and
runtime. Users must comply with Rubber Band's license when building, linking,
distributing, or deploying software that uses `rubband`.

### Known limitations

- Audio input must be CPU, contiguous, `float32` memory.
- GPU tensors are not supported.
- Source builds require Rubber Band 4.x to be installed separately.
- `LiveShifter` follows Rubber Band's live API, but Python itself is not a
  hard real-time runtime.
- Python logger callbacks are diagnostic only and are not hard real-time safe.
- Dynamic ratio changes are for real-time `Stretcher` use. Offline stretchers
  reject ratio changes after `study()` or `process()` starts.

### Local release checklist

Before tagging a public release, run:

```sh
uv run ruff check --fix --select B,E,F,I rubband tests scripts
uv run ruff format rubband tests scripts
uv run ty check rubband
find tests rubband scripts -name '*.py' | xargs uv run pyupgrade --py311-plus
uv run pytest
uv run mkdocs build --strict
uv build --sdist --wheel --out-dir dist
uv run python scripts/smoke_wheel.py dist/rubband-*.whl
```

Then confirm the release workflow is green on macOS, Linux, and Windows after
pushing the release tag, and inspect the platform artifacts for repaired
wheels in `wheelhouse`.

### Initial release highlights

- Python 3.11+.
- Rubber Band-backed time stretching and pitch shifting.
- Stateful `Stretcher` API close to Rubber Band's original model.
- Rubber Band 4.x `LiveShifter` API for fixed-block live pitch shifting.
- DLPack and Python buffer protocol input support.
- NumPy is not required at runtime.
- PyTorch CPU tensors are supported when PyTorch is installed.
- `AudioBuffer` output wrapper with `memoryview()`, `numpy()`, and `torch()`
  conversion helpers.
