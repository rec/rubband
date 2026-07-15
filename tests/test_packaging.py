from __future__ import annotations

from pathlib import Path


def test_release_workflow_builds_platform_wheels() -> None:
    workflow = Path(".github/workflows/release-builds.yml").read_text()

    assert "git merge-base --is-ancestor" in workflow
    assert "macos-latest" in workflow
    assert "ubuntu-22.04" in workflow
    assert "windows-latest" in workflow
    assert "brew install rubberband pkg-config" in workflow
    assert (
        "sudo apt-get install -y build-essential curl meson ninja-build pkg-config"
        in workflow
    )
    assert "rubberband/archive/refs/tags/v4.0.0.tar.gz" in workflow
    assert "sudo ninja -C build install" in workflow
    assert "ilammy/msvc-dev-cmd@v1" in workflow
    assert "choco install ninja --no-progress" in workflow
    assert "vcpkg install rubberband:x64-windows" in workflow
    assert "CMAKE_GENERATOR=Ninja" in workflow
    assert "uv sync --dev --no-editable" in workflow
    assert "uv build --sdist --wheel --out-dir dist" in workflow
    assert "scripts/smoke_wheel.py" in workflow
    assert "gh release create" in workflow


def test_release_script_builds_and_smokes_wheel() -> None:
    script = Path("scripts/release.sh").read_text()

    assert "uv version --bump" in script
    assert "uv build --sdist --wheel --out-dir dist" in script
    assert "scripts/smoke_wheel.py" in script
    assert "git tag" in script


def test_scikit_build_uses_persistent_build_dir() -> None:
    pyproject = Path("pyproject.toml").read_text()

    assert 'build-dir = "build/scikit-build/{state}/{wheel_tag}"' in pyproject


def test_cmake_has_non_pkg_config_rubber_band_fallback() -> None:
    cmake = Path("CMakeLists.txt").read_text()

    assert "nanobind_add_module(_rubband rubband/_rubband.cpp)" in cmake
    assert "pkg_check_modules(RUBBERBAND QUIET IMPORTED_TARGET rubberband)" in cmake
    assert "find_path(RUBBERBAND_INCLUDE_DIR rubberband/RubberBandStretcher.h)" in cmake
    assert "find_library(RUBBERBAND_LIBRARY" in cmake
    assert "RUBBERBAND_API_MAJOR_VERSION < 3" in cmake
    assert "Rubband requires Rubber Band API 3.0 or newer" in cmake
    assert "CMAKE_PREFIX_PATH, or vcpkg" in cmake


def test_api_docs_do_not_expose_native_internals() -> None:
    api = Path("docs/api.md").read_text()

    assert "rubband._native" not in api
    assert "rubband._rubband" not in api
    assert "stretch_float32" not in api


def test_pyinstaller_entrypoint_exists_for_release_script() -> None:
    entrypoint = Path("install/pyinstaller_entrypoint.py")

    assert entrypoint.is_file()
    text = entrypoint.read_text()
    assert "rubband._rubband" in text
    assert "rubband_main" in text
