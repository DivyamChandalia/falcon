#!/usr/bin/env python3
"""Capture both README Resources images from the current cluster state."""

from __future__ import annotations

import asyncio
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.pop("NO_COLOR", None)

from falcon.config import load_config  # noqa: E402
from falcon.resources import fetch_cluster_snapshot  # noqa: E402
from falcon.resources_history import history_store  # noqa: E402
from falcon.resources_ui import FalconResourcesApp  # noqa: E402

NODES_ASSET = ROOT / "assets" / "falcon-resources.svg"
ALLOCATIONS_ASSET = ROOT / "assets" / "falcon-resources-allocations.svg"


class SnapshotCollector:
    """Serve one immutable live frame while Textual composes the screenshot."""

    def __init__(self, snapshot) -> None:
        self.snapshot = snapshot

    def collect(self, force: bool = False):
        del force
        return self.snapshot

    def close(self) -> None:
        pass


async def capture() -> None:
    config = load_config()
    snapshot = fetch_cluster_snapshot(
        str(config["cluster"]["kube_state_metrics_url"]),
        timeout=10,
        collected_at=time.time(),
    )
    if not snapshot.nodes:
        raise RuntimeError("current cluster snapshot contains no nodes")

    store = history_store(config)
    app = FalconResourcesApp(
        SnapshotCollector(snapshot),
        refresh_seconds=3600,
        history_loader=store.load,
        history_hours=float(
            config.get("resources", {}).get("history_hours", 24)
        ),
        initial_view="nodes",
    )
    async with app.run_test(size=(140, 32)) as pilot:
        await pilot.pause(0.5)
        nodes_svg = app.export_screenshot(
            title="Falcon resources · live cluster snapshot",
            simplify=True,
        )

        await pilot.press("right")
        await pilot.pause(0.5)
        allocations_svg = app.export_screenshot(
            title="Falcon GPU allocations · live cluster snapshot",
            simplify=True,
        )

    NODES_ASSET.write_text(nodes_svg, encoding="utf-8")
    ALLOCATIONS_ASSET.write_text(allocations_svg, encoding="utf-8")
    availability = ", ".join(
        f"{item.model} {item.request_headroom}/{item.allocatable}"
        for item in snapshot.gpu_availability.values()
    )
    print(
        f"captured {len(snapshot.nodes)} live nodes to {NODES_ASSET}\n"
        f"captured live GPU allocations to {ALLOCATIONS_ASSET}\n"
        f"GPU request headroom: {availability or '-'}"
    )


if __name__ == "__main__":
    asyncio.run(capture())
