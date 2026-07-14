from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from shutil import which
from subprocess import run

import numpy as np


def smoke_installed_package() -> None:
    import rubband

    audio = np.zeros(48_000, dtype=np.float32)
    out = rubband.stretch(audio, rubband.StretchOptions(sample_rate=48_000))
    assert out.dtype == np.float32
    assert out.ndim == 1
    assert out.shape[0] >= 48_000


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
