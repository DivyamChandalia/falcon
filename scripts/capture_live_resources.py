#!/usr/bin/env python3
"""Capture the README resources image from the current cluster snapshot."""

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
from falcon.resources_ui import FalconResourcesApp  # noqa: E402

ASSET = ROOT / "assets" / "falcon-resources.svg"


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

    app = FalconResourcesApp(
        SnapshotCollector(snapshot),
        refresh_seconds=3600,
    )
    async with app.run_test(size=(140, 32)) as pilot:
        await pilot.pause(0.5)
        # Prefer a quiet node so the public README demonstrates real headroom
        # without publishing current user workload names.
        quiet = next(
            (
                node
                for node in app.nodes
                if not node.visible_consumers
                and node.schedulable
                and node.allocatable.gpu_count
            ),
            None,
        ) or next(
            (node for node in app.nodes if not node.visible_consumers),
            None,
        )
        if quiet is not None:
            app.state.selected_node = quiet.name
            app._ensure_visible()
            app._render_all()
            await pilot.pause()
        svg = app.export_screenshot(
            title="Falcon resources · live cluster snapshot",
            simplify=True,
        )

    ASSET.write_text(svg, encoding="utf-8")
    availability = ", ".join(
        f"{item.model} {item.request_headroom}/{item.allocatable}"
        for item in snapshot.gpu_availability.values()
    )
    print(
        f"captured {len(snapshot.nodes)} live nodes to {ASSET}\n"
        f"GPU request headroom: {availability or '-'}"
    )


if __name__ == "__main__":
    asyncio.run(capture())
