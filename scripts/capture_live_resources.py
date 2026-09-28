#!/usr/bin/env python3
"""Capture the README dashboards from the current cluster state."""

from __future__ import annotations

import asyncio
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.pop("NO_COLOR", None)

from falcon.cluster import ClusterCollector  # noqa: E402
from falcon.config import load_config  # noqa: E402
from falcon.dashboard import UsageCollector  # noqa: E402
from falcon.dashboard_logs import DashboardLogManager  # noqa: E402
from falcon.dashboard_ui import FalconDashboard  # noqa: E402
from falcon.kubernetes import KubernetesClient  # noqa: E402
from falcon.resource_service import ResourceServiceClient  # noqa: E402
from falcon.resources import fetch_cluster_snapshot  # noqa: E402
from falcon.resources_history import history_store  # noqa: E402
from falcon.resources_ui import FalconResourcesApp  # noqa: E402

JOBS_ASSET = ROOT / "assets" / "falcon-dashboard.svg"
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
    namespace = str(config["cluster"]["namespace"])

    dashboard_config = config.get("dashboard", {})
    thresholds = {
        str(preset["gpu_type"]).lower(): float(
            preset.get("minimum_utilization", 30)
        )
        for preset in config.get("presets", {}).values()
    }
    # Use the same direct Kubernetes inventory path as the CLI fallback.  The
    # configured resource-service hostname may be unavailable from the capture
    # host even when kubectl has access to the current namespace.
    dashboard_collector = UsageCollector(
        namespace,
        thresholds,
        float(dashboard_config.get("ema_alpha", 0.1)),
        streaming_gpu=False,
        collect_availability=False,
    )
    initial_jobs = dashboard_collector.collect()
    log_manager = DashboardLogManager(namespace)
    dashboard = FalconDashboard(
        dashboard_collector,
        refresh_seconds=3600,
        sort_field=str(dashboard_config.get("sort_field", "Age")),
        sort_direction=str(dashboard_config.get("sort_direction", "desc")),
        log_manager=log_manager,
        launch_config=config,
    )
    async with dashboard.run_test(size=(200, 50)) as pilot:
        await pilot.pause(0.5)
        if not dashboard.rows:
            dashboard.rows = list(initial_jobs)
            dashboard._filter_rows()
            log_manager.reconcile(dashboard.rows)
            dashboard._render_all()
        # Give the live dashboard a full refresh interval to establish the
        # selected Pod's attach stream and render useful workload output.
        await pilot.pause(60.0)
        running_pcvit = next(
            (
                row
                for row in dashboard.rows
                if row.status.lower() == "running"
                and row.job == "pcvit-rgb-dinov3b-400e"
            ),
            None,
        )
        if running_pcvit is None:
            running_pcvit = next(
                (
                    row
                    for row in dashboard.rows
                    if row.status.lower() == "running"
                    and row.job.lower().startswith("pcvit-")
                ),
                None,
            )
        if running_pcvit is not None:
            dashboard.state.cursor_job_uid = running_pcvit.uid
            dashboard._selection_changed()
            await pilot.pause(1.0)
        # Keep the complete large dashboard visible while making the running
        # pcvit workload the active selection. The selected-job inspector and
        # its live Logs remain visible in the right column.
        await pilot.press("4")
        await pilot.pause(0.2)
        await pilot.pause(0.8)
        dashboard_svg = dashboard.export_screenshot(
            title="Falcon dashboard · live jobs and logs",
            simplify=True,
        )

    try:
        snapshot = fetch_cluster_snapshot(
            str(config["cluster"]["kube_state_metrics_url"]),
            timeout=10,
            collected_at=time.time(),
        )
    except Exception:
        # Prefer the service's last-known-good local cache when its hostname
        # is unavailable from the capture host. This is the same fallback the
        # human-facing resources command uses and preserves a real cluster
        # frame instead of substituting demo data.
        service_url = config.get("cluster", {}).get("resource_service_url")
        if service_url:
            snapshot = ResourceServiceClient(str(service_url)).snapshot()
        else:
            snapshot = None
        if snapshot is None or not snapshot.nodes:
            # Keep the capture useful on hosts where both metrics DNS and the
            # service cache are unavailable but kubectl can read inventory.
            direct_collector = ClusterCollector(
                KubernetesClient(namespace),
                namespace="",
                inventory_seconds=60,
            )
            snapshot = direct_collector.collect(force=True)
            direct_collector.close()
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
    async with app.run_test(size=(200, 50)) as pilot:
        await pilot.pause(0.5)
        if any(node.name == "nodex1" for node in app.nodes):
            app.state.selected_node = "nodex1"
            app.state.selected_consumer = 0
            app.state.consumer_scroll = 0
            app._render_all()
        nodes_svg = app.export_screenshot(
            title="Falcon resources · live cluster snapshot",
            simplify=True,
        )

        await pilot.press("right")
        await pilot.pause(0.5)
        allocations_svg = app.export_screenshot(
            title="Falcon allocations · live cluster snapshot",
            simplify=True,
        )

    JOBS_ASSET.write_text(dashboard_svg, encoding="utf-8")
    NODES_ASSET.write_text(nodes_svg, encoding="utf-8")
    ALLOCATIONS_ASSET.write_text(allocations_svg, encoding="utf-8")
    availability = ", ".join(
        f"{item.model} {item.request_headroom}/{item.allocatable}"
        for item in snapshot.gpu_availability.values()
    )
    selected_job = running_pcvit.job if running_pcvit is not None else "-"
    print(
        f"captured live jobs and logs to {JOBS_ASSET} (selected {selected_job})\n"
        f"captured {len(snapshot.nodes)} live nodes to {NODES_ASSET}\n"
        f"captured live allocations to {ALLOCATIONS_ASSET}\n"
        f"GPU request headroom: {availability or '-'}"
    )


if __name__ == "__main__":
    asyncio.run(capture())
