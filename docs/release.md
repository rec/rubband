# Release

Release builds are tag-driven. Pushing a `v*` tag runs the GitHub Actions
release workflow.

The workflow:

1. Verifies the tagged commit is on `main`.
2. Builds wheels and a source distribution on macOS, Linux, and Windows.
3. Smoke-tests the wheel.
4. Packages artifacts and SHA256 checksums.
5. Publishes a GitHub release.

Local release preparation should include:

```sh
uv run pytest
uv run mkdocs build --strict
uv build --sdist --wheel --out-dir dist
```
