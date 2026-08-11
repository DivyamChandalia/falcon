#!/usr/bin/env python3
"""Generate deterministic Falcon TUI goldens and human-review SVGs."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
# Review artifacts must preserve Falcon's truecolor visual semantics even when
# an automation host sets NO_COLOR for ordinary command output.
os.environ.pop("NO_COLOR", None)

from falcon.dashboard import DemoUsageCollector  # noqa: E402
from falcon.dashboard_ui import FalconDashboard  # noqa: E402
from falcon.demo import DEMO_NOW, DemoCollector  # noqa: E402
from falcon.resources_charts import GPUHistoryPoint  # noqa: E402
from falcon.resources_ui import FalconResourcesApp  # noqa: E402

ARTIFACTS = ROOT / "artifacts" / "tui"
SNAPSHOTS = ROOT / "tests" / "snapshots"
ASSETS = ROOT / "assets"
DIMENSIONS: Tuple[Tuple[int, int], ...] = (
    (60, 18),
    (79, 21),
    (80, 22),
    (80, 24),
    (80, 30),
    (90, 22),
    (100, 24),
    (100, 30),
    (120, 30),
    (140, 32),
    (160, 40),
    (200, 50),
)


def normalize_svg(value: str) -> str:
    """Remove Rich's per-console identifier while retaining every visual cell."""

    value = re.sub(r"terminal-\d+", "terminal-ID", value)
    value = re.sub(r"\d{2}:\d{2}:\d{2}", "12:00:00", value)
    return value.replace("\r\n", "\n")


def digest(value: str) -> str:
    return hashlib.sha256(normalize_svg(value).encode("utf-8")).hexdigest()


def write_svg(name: str, value: str) -> None:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    (ARTIFACTS / f"{name}.svg").write_text(
        normalize_svg(value),
        encoding="utf-8",
    )


async def dashboard_capture(
    name: str,
    *,
    state: str = "mixed",
    size: Tuple[int, int] = (120, 30),
    actions=(),
) -> Tuple[str, str]:
    app = FalconDashboard(
        DemoUsageCollector(state),
        refresh_seconds=999,
        clock=lambda: "12:00:00",
    )
    async with app.run_test(size=size) as pilot:
        await pilot.pause(0.5)
        for action in actions:
            if callable(action):
                action(app)
                await pilot.pause()
            else:
                await pilot.press(*str(action).split())
        # Spinner cadence is intentionally live in the product, but it is not
        # visual state. Freeze it before hashing so slower CI machines cannot
        # produce a different golden for the same layout.
        app._spinner = 0
        app._render_header()
        value = app.export_screenshot(
            title=f"Falcon dashboard · {name}", simplify=True
        )
    write_svg(name, value)
    return name, digest(value)


async def resources_capture(
    name: str,
    *,
    state: str = "mixed",
    size: Tuple[int, int] = (120, 30),
    actions=(),
) -> Tuple[str, str]:
    app = FalconResourcesApp(
        DemoCollector(state),
        refresh_seconds=999,
        clock=lambda snapshot: "12:00:00",
    )
    async with app.run_test(size=size) as pilot:
        await pilot.pause(0.5)
        for action in actions:
            if callable(action):
                action(app)
                await pilot.pause()
            else:
                await pilot.press(*str(action).split())
        app._spinner = 0
        app._render_header()
        value = app.export_screenshot(
            title=f"Falcon resources · {name}", simplify=True
        )
    write_svg(name, value)
    return name, digest(value)


def search_long(app: FalconDashboard) -> None:
    app.state.search_query = "extraordinarily"
    app._filter_rows()
    app._render_all()


def active_filter(app: FalconDashboard) -> None:
    app.state.filters["status"] = "Running"
    app._filter_rows()
    app._render_all()


def filter_dialog(app: FalconDashboard) -> None:
    app.action_filters()


def action_dialog(app: FalconDashboard) -> None:
    app.action_kill()


def seed_resources_history(app: FalconResourcesApp) -> None:
    """Populate a deterministic since-launch window for visual review."""

    namespaces = sorted(
        {
            consumer.namespace
            for node in app._gpu_nodes()
            if node.ready and node.schedulable
            for consumer in node.consumers
            if consumer.requested.gpu_count > 0
        }
    )
    app.history = []
    for index in range(16):
        values = {
                namespace: min(
                    8,
                    (index // (namespace_index + 2) + namespace_index)
                    % 9,
                )
                for namespace_index, namespace in enumerate(namespaces)
            }
        app.history.append(
            GPUHistoryPoint.from_mapping(
                DEMO_NOW - (15 - index) * 5 * 60,
                values,
                {namespace: value * 80 for namespace, value in values.items()},
            )
        )
    app._render_all()


async def capture() -> None:
    results: Dict[str, str] = {}

    for width, height in DIMENSIONS:
        name, value = await dashboard_capture(
            f"dashboard-many-{width}x{height}",
            state="many",
            size=(width, height),
        )
        results[name] = value

    dashboard_states = (
        ("dashboard-default", "mixed", (120, 30), ()),
        ("dashboard-selected-job", "mixed", (120, 30), ("4",)),
        ("dashboard-expanded-jobs", "many", (120, 30), ("enter",)),
        ("dashboard-expanded-selected", "mixed", (120, 30), ("4", "enter")),
        ("dashboard-expanded-resources", "mixed", (140, 32), ("2", "enter")),
        ("dashboard-expanded-events", "mixed", (120, 30), ("3", "enter")),
        ("dashboard-events-middle", "mixed", (80, 30), ("3", "home", "pagedown")),
        ("dashboard-search", "mixed", (100, 24), (search_long,)),
        ("dashboard-filters", "mixed", (100, 24), (active_filter,)),
        ("dashboard-filter-dialog", "mixed", (100, 24), (filter_dialog,)),
        ("dashboard-action-dialog", "mixed", (100, 24), (action_dialog,)),
        ("dashboard-stale", "stale", (100, 24), ()),
        ("dashboard-no-jobs", "no-jobs", (100, 24), ()),
        ("dashboard-long-content", "mixed", (80, 22), (search_long, "4", "enter")),
    )
    for name, state, size, actions in dashboard_states:
        key, value = await dashboard_capture(
            name, state=state, size=size, actions=actions
        )
        results[key] = value

    resource_states = (
        ("resources-80x22", "mixed", (80, 22), ()),
        ("resources-140x32", "mixed", (140, 32), ()),
        ("resources-200x50", "mixed", (200, 50), ()),
        (
            "resources-gpu-allocations-80x22",
            "mixed",
            (80, 22),
            (seed_resources_history, "right"),
        ),
        (
            "resources-gpu-allocations-140x32",
            "mixed",
            (140, 32),
            (seed_resources_history, "right"),
        ),
        (
            "resources-gpu-allocations-200x50",
            "mixed",
            (200, 50),
            (seed_resources_history, "right"),
        ),
        (
            "resources-gpu-allocations-vram-140x32",
            "mixed",
            (140, 32),
            (seed_resources_history, "right", "v"),
        ),
        ("resources-node-expanded-80x22", "mixed", (80, 22), ("enter",)),
        ("resources-node-expanded-140x40", "mixed", (140, 40), ("enter",)),
        ("resources-stale", "stale", (120, 30), ()),
        ("resources-no-jobs", "no-jobs", (120, 30), ()),
        (
            "resources-gpu-allocations-stale",
            "stale",
            (140, 32),
            (seed_resources_history, "right"),
        ),
        (
            "resources-gpu-allocations-no-gpus",
            "no-gpus",
            (140, 32),
            ("right",),
        ),
    )
    for name, state, size, actions in resource_states:
        key, value = await resources_capture(
            name, state=state, size=size, actions=actions
        )
        results[key] = value

    # Keep the README visuals on the same deterministic 140×32 canvas so their
    # typography, gutters, and section borders can be compared directly.
    dashboard_asset, dashboard_digest = await dashboard_capture(
        "dashboard-asset", state="mixed", size=(140, 32)
    )
    resources_asset, resources_digest = await resources_capture(
        "resources-asset", state="mixed", size=(140, 32)
    )
    allocations_asset, allocations_digest = await resources_capture(
        "resources-allocations-asset",
        state="mixed",
        size=(140, 32),
        actions=(seed_resources_history, "right"),
    )
    results[dashboard_asset] = dashboard_digest
    results[resources_asset] = resources_digest
    results[allocations_asset] = allocations_digest

    SNAPSHOTS.mkdir(parents=True, exist_ok=True)
    manifest = {
        "format": 1,
        "textual": "7.x",
        "dimensions": [f"{width}x{height}" for width, height in DIMENSIONS],
        "snapshots": dict(sorted(results.items())),
    }
    (SNAPSHOTS / "tui_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    ASSETS.mkdir(parents=True, exist_ok=True)
    (ASSETS / "falcon-dashboard.svg").write_text(
        (ARTIFACTS / "dashboard-asset.svg").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    (ASSETS / "falcon-resources.svg").write_text(
        (ARTIFACTS / "resources-asset.svg").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    (ASSETS / "falcon-resources-allocations.svg").write_text(
        (ARTIFACTS / "resources-allocations-asset.svg").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    print(
        f"captured {len(results)} deterministic states in {ARTIFACTS}\n"
        f"golden manifest: {SNAPSHOTS / 'tui_manifest.json'}"
    )


if __name__ == "__main__":
    asyncio.run(capture())
