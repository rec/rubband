# Building

`rubband` contains a nanobind extension over Rubber Band. Building from source
requires Rubber Band development headers and libraries.

## macOS

```sh
brew install rubberband pkg-config
uv build --sdist --wheel --out-dir dist
```

## Linux

Install your distribution's Rubber Band development package and `pkg-config`.
On Debian or Ubuntu:

```sh
sudo apt-get update
sudo apt-get install -y librubberband-dev pkg-config
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
