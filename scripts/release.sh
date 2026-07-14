#!/usr/bin/env bash
set -euxo pipefail

part="${1:-patch}"

if [[ "$part" != "patch" && "$part" != "minor" && "$part" != "major" ]]; then
  echo "usage: scripts/release.sh [patch|minor|major]" >&2
  exit 2
fi

if [[ -n "$(git status --porcelain)" ]]; then
  echo "release requires a clean working tree" >&2
  exit 1
fi

branch="$(git branch --show-current)"
if [[ "$branch" != "main" ]]; then
  echo "release must run from main, not $branch" >&2
  exit 1
fi

target_version="$(uv version --bump "$part" --dry-run --short)"
tag="v$target_version"

if git rev-parse --verify --quiet "$tag" >/dev/null; then
  echo "tag already exists: $tag" >&2
  exit 1
fi

if git ls-remote --exit-code --tags origin "refs/tags/$tag" >/dev/null; then
  echo "remote tag already exists: $tag" >&2
  exit 1
fi

uv sync --dev
uv build --wheel --no-sources --out-dir dist
wheel="$(find dist -maxdepth 1 -name 'rubband-*.whl' | sort | tail -n 1)"
uv pip install --reinstall "$wheel"

.venv/bin/pytest
.venv/bin/pytest --run-long tests/test_native.py::test_native_pianolead_pitch_regression
.venv/bin/ruff check --fix --select B,E,F,I src tests scripts
.venv/bin/ruff format src tests scripts
.venv/bin/ty check src/rubband
find tests src scripts -name '*.py' | xargs .venv/bin/pyupgrade --py313-plus
git diff --check

if [[ -n "$(git status --porcelain)" ]]; then
  echo "release verification changed the working tree" >&2
  exit 1
fi

uv build --sdist --wheel --out-dir dist
wheel="$(find dist -maxdepth 1 -name 'rubband-*.whl' | sort | tail -n 1)"
.venv/bin/python scripts/smoke_wheel.py "$wheel"

uv version --bump "$part" --no-sync
version="$(uv version --short)"
tag="v$version"

if [[ "$version" != "$target_version" ]]; then
  echo "version changed after dry run: expected $target_version, got $version" >&2
  exit 1
fi

uv lock
git add pyproject.toml uv.lock
git commit -m "Release $tag"
git push
git tag "$tag"
git push origin "$tag"
