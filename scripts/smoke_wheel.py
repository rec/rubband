from __future__ import annotations

import sys
import tempfile
from array import array
from pathlib import Path
from shutil import which
from subprocess import run


def smoke_installed_package() -> None:
    import rubband

    audio = array("f", [0.0] * 48_000)
    out = rubband.stretch(audio, 48_000)
    assert out.dtype == "float32"
    assert out.shape[0] >= 48_000
    view = memoryview(out.memoryview())  # ty: ignore[invalid-argument-type]
    assert view.format == "f"


def main() -> None:
    if sys.argv == [sys.argv[0], "--inside-venv"]:
        smoke_installed_package()
        return

    if len(sys.argv) != 2:
        sys.exit("usage: smoke_wheel.py WHEEL")

    wheel = Path(sys.argv[1]).resolve()
    if not wheel.exists():
        sys.exit(f"wheel does not exist: {wheel}")
    if which("uv") is None:
        sys.exit("uv is required to create the smoke-test environment")

    with tempfile.TemporaryDirectory(prefix="rubband-wheel-") as directory:
        environment = Path(directory)
        run(["uv", "venv", "--seed", str(environment)], check=True)
        if sys.platform == "win32":
            python = environment / "Scripts/python.exe"
        else:
            python = environment / "bin/python"
        run(["uv", "pip", "install", "--python", str(python), str(wheel)], check=True)
        run([python, __file__, "--inside-venv"], check=True)


if __name__ == "__main__":
    main()
