# Development and release checks

The primary suite uses standard `unittest` and is also pytest-compatible:

```console
python -m unittest discover -s tests -v
python -m pytest
```

Visual fixtures and reviewable SVGs:

```console
python scripts/capture_tui_snapshots.py
python scripts/capture_live_resources.py
```

The matrix covers 60×18, 79×21, 80×22, 80×24, 80×30, 90×22,
100×24, 100×30, 120×30, 140×32, 160×40, and 200×50, plus navigation,
expanded panes, filters/dialogs, stale state, long content, and node drill-in.
The deterministic capture updates the review goldens on a shared 140×32
canvas. The live Resources capture additionally requires the configured
metrics endpoint and replaces both Resources README assets with the actual
cluster snapshot and persisted allocation history.

Release validation:

```console
python -m compileall -q falcon tests
python -m unittest discover -s tests -v
python -m build
python -m venv /tmp/falcon-wheel-test
/tmp/falcon-wheel-test/bin/pip install dist/falcon_k8s-*.whl
/tmp/falcon-wheel-test/bin/falcon --version
/tmp/falcon-wheel-test/bin/falcon --help
```

Versioning follows semantic `MAJOR.MINOR.PATCH` releases. The single source
of truth is `falcon/__init__.py`; `pyproject.toml` reads that value dynamically
when building wheels, so the CLI and package metadata cannot drift. For a
release, update that value, add a matching `vMAJOR.MINOR.PATCH` Git tag, run
the wheel checks above, and push the commit and tag together. The daily update
check compares this metadata with the `main` branch before prompting.

Confirm the wheel contains only `falcon`, bundled skills, and required
metadata; it must not install another console entrypoint. Optional kind tests
require explicit `FALCON_KIND_INTEGRATION=1` and a current `kind-*` context.

CI runs supported Python versions, lint, tests, visual regression checks, and
the clean wheel-install smoke test.
