# Release

Release builds are tag-driven. Pushing a `v*` tag runs the GitHub Actions
release workflow.

The workflow:

1. Verifies the tagged commit is on `main`.
2. Builds wheels and a source distribution on macOS, Linux, and Windows.
3. Repairs each wheel so the Rubber Band 4.x native libraries are bundled.
4. Smoke-tests the repaired wheel.
5. Packages artifacts and SHA256 checksums.
6. Publishes a GitHub release.

Local release preparation should include:

```sh
uv run pytest
uv run mkdocs build --strict
uv build --sdist --wheel --out-dir dist
```
