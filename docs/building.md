# Building

`rubband` contains a nanobind extension over Rubber Band. Binary release
wheels bundle Rubber Band 4.x, but building from source requires Rubber Band
4.x development headers and libraries.

## macOS

```sh
brew install rubberband pkg-config
uv build --sdist --wheel --out-dir dist
```

## Linux

Rubber Band 4.0 or newer is required. Many stable distributions still package
older Rubber Band headers, so build and install Rubber Band 4.x first if your
distribution package is too old:

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
uv build --sdist --wheel --out-dir dist
```

## Windows

Install Rubber Band with `vcpkg`, then pass the vcpkg CMake toolchain file when
building:

```powershell
vcpkg install rubberband:x64-windows
$env:CMAKE_ARGS="-DCMAKE_TOOLCHAIN_FILE=$env:VCPKG_INSTALLATION_ROOT/scripts/buildsystems/vcpkg.cmake"
uv build --sdist --wheel --out-dir dist
```

## Smoke Test

```sh
uv run python scripts/smoke_wheel.py dist/rubband-*.whl
```
