from __future__ import annotations

from pathlib import Path


def test_release_workflow_builds_platform_wheels() -> None:
    workflow = Path(".github/workflows/release-builds.yml").read_text()

    assert "git merge-base --is-ancestor" in workflow
    assert "macos-latest" in workflow
    assert "ubuntu-22.04" in workflow
    assert "windows-latest" in workflow
    assert "brew install rubberband pkg-config" in workflow
    assert "sudo apt-get install -y librubberband-dev pkg-config" in workflow
    assert "vcpkg install rubberband:x64-windows" in workflow
    assert "uv build --sdist --wheel --out-dir dist" in workflow
    assert "scripts/smoke_wheel.py" in workflow
    assert "gh release create" in workflow


def test_release_script_builds_and_smokes_wheel() -> None:
    script = Path("scripts/release.sh").read_text()

    assert "uv version --bump" in script
    assert "uv build --sdist --wheel --out-dir dist" in script
    assert "scripts/smoke_wheel.py" in script
    assert "git tag" in script


def test_cmake_has_non_pkg_config_rubber_band_fallback() -> None:
    cmake = Path("CMakeLists.txt").read_text()

    assert "pkg_check_modules(RUBBERBAND QUIET IMPORTED_TARGET rubberband)" in cmake
    assert "find_path(RUBBERBAND_INCLUDE_DIR rubberband/RubberBandStretcher.h)" in cmake
    assert "find_library(RUBBERBAND_LIBRARY" in cmake
    assert "CMAKE_PREFIX_PATH, or vcpkg" in cmake
