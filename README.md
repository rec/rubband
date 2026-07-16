# ➰ `rubband`: tensor-friendly bindings to the C++ Rubber Band pitch shifting library ➰

## What is this?

Rubber Band is a popular GPL C++ library for pitch shifting and time stretching with a
long history, but all the Python bindings for it are old and unsupported.

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

## If you're human, why is there an emoji in the description of this project??

I've been using emojis in project descriptions for about six years. They learned it from me.

#### Nothing below this line was written by a person.

------


## Public release notes

The first public release is intended to be a practical, minimal binding to the
Rubber Band stretcher. The Python package wraps the native Rubber Band library;
it does not bundle Rubber Band itself.

### Installation

Install the platform Rubber Band library first, then install `rubband`.

On macOS:

```sh
brew install rubberband pkg-config
pip install rubband
```

On Linux, Rubber Band API 3.0 or newer is required. Many stable
distributions still package older Rubber Band headers, so build and install
Rubber Band 4.x first if your distribution package is too old:

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

### Platform support

Release builds are configured for:

- macOS
- Linux
- Windows

Each platform wheel is built and smoke-tested in GitHub Actions against the
Rubber Band library installed by that platform's package manager or `vcpkg`.
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
terms. Users must have the Rubber Band library available at build and runtime
and must comply with Rubber Band's license when building, linking,
distributing, or deploying software that uses `rubband`.

### Known limitations

- Audio input must be CPU, contiguous, `float32` memory.
- GPU tensors are not supported.
- The Rubber Band native library must be installed separately.
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
pushing the release tag.

### Initial release highlights

- Python 3.11+.
- Rubber Band-backed time stretching and pitch shifting.
- Stateful `Stretcher` API close to Rubber Band's original model.
- DLPack and Python buffer protocol input support.
- NumPy is not required at runtime.
- PyTorch CPU tensors are supported when PyTorch is installed.
- `AudioBuffer` output wrapper with `memoryview()`, `numpy()`, and `torch()`
  conversion helpers.
