from __future__ import annotations

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--run-long",
        action="store_true",
        default=False,
        help="run long audio regression tests",
    )


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "long: long audio regression test")


def pytest_collection_modifyitems(
    config: pytest.Config,
    items: list[pytest.Item],
) -> None:
    if config.getoption("--run-long"):
        return
    skip_long = pytest.mark.skip(reason="need --run-long option to run")
    for item in items:
        if "long" in item.keywords:
            item.add_marker(skip_long)
