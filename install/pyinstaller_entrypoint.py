from __future__ import annotations

import sys
from pathlib import Path

SOURCE_ROOT = Path(__file__).resolve().parents[1]
if not getattr(sys, "frozen", False) and str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

import rubband._rubband  # noqa: F401, E402
from rubband import main as rubband_main  # noqa: E402


def main() -> None:
    rubband_main()


if __name__ == "__main__":
    main()
