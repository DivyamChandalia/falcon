"""Scheduler allocation accounting for the Resources GPU page.

The Resources endpoint exposes Kubernetes allocation/request data, not device
utilization.  Keep that distinction explicit: the two selectable bases are
GPU-count allocation and allocated VRAM (GPU request count multiplied by the
node's per-device VRAM label).
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, replace
from typing import Mapping, Sequence

from .cluster import NodeSnapshot, natural_name_key


@dataclass(frozen=True)
class GpuTelemetrySnapshot:
    """One scheduler allocation observation from a resource snapshot."""

    collected_at: float = 0.0
    effective_gpus_by_node: tuple[tuple[str, float], ...] = ()
    effective_gpus_by_namespace: tuple[tuple[str, float], ...] = ()
    vram_gib_by_namespace: tuple[tuple[str, float], ...] = ()
    target_pods: int = 0
    sampled_pods: int = 0
    stale: bool = False
    error: str = ""

    @staticmethod
    def _ordered(values: Mapping[str, float]) -> tuple[tuple[str, float], ...]:
        return tuple(
            sorted(
                (
                    (str(name), max(0.0, float(value)))
                    for name, value in values.items()
                    if value >= 0
                ),
                key=lambda item: natural_name_key(item[0]),
            )
        )

    @classmethod
    def from_mappings(
        cls,
        *,
        collected_at: float,
        effective_gpus_by_node: Mapping[str, float],
        effective_gpus_by_namespace: Mapping[str, float],
        vram_gib_by_namespace: Mapping[str, float],
        target_pods: int,
        sampled_pods: int,
    ) -> "GpuTelemetrySnapshot":
        return cls(
            collected_at=float(collected_at),
            effective_gpus_by_node=cls._ordered(effective_gpus_by_node),
            effective_gpus_by_namespace=cls._ordered(
                effective_gpus_by_namespace
            ),
            vram_gib_by_namespace=cls._ordered(vram_gib_by_namespace),
            target_pods=max(0, int(target_pods)),
            sampled_pods=max(0, int(sampled_pods)),
        )

    def mark_stale(self, error: str) -> "GpuTelemetrySnapshot":
        return replace(self, stale=True, error=str(error))


def allocation_snapshot(
    nodes: Sequence[NodeSnapshot],
    *,
    collected_at: float = 0.0,
    stale: bool = False,
    error: str = "",
) -> GpuTelemetrySnapshot:
    """Calculate GPU-count and allocated-VRAM shares from node consumers.

    This is intentionally based only on the same scheduler-facing snapshot
    used by ``falcon resources``.  It never claims to measure GPU compute or
    physical memory occupancy.
    """

    by_node: defaultdict[str, float] = defaultdict(float)
    by_namespace: defaultdict[str, float] = defaultdict(float)
    vram_by_namespace: defaultdict[str, float] = defaultdict(float)
    pod_count = 0
    for node in nodes:
        if node.ready is not True or not node.schedulable:
            continue
        per_device_gib = (
            node.gpu_memory_bytes_per_device / (1024**3)
            if node.gpu_memory_bytes_per_device is not None
            else 0.0
        )
        for consumer in node.consumers:
            count = max(0, int(consumer.requested.gpu_count))
            if count <= 0:
                continue
            pod_count += 1
            by_node[node.name] += count
            by_namespace[consumer.namespace] += count
            if per_device_gib > 0:
                vram_by_namespace[consumer.namespace] += count * per_device_gib
    timestamp = float(collected_at) or time.time()
    snapshot = GpuTelemetrySnapshot.from_mappings(
        collected_at=timestamp,
        effective_gpus_by_node=by_node,
        effective_gpus_by_namespace=by_namespace,
        vram_gib_by_namespace=vram_by_namespace,
        target_pods=pod_count,
        sampled_pods=pod_count,
    )
    return replace(snapshot, stale=stale, error=error)
