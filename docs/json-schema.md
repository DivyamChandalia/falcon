# Machine-readable output

Falcon emits one JSON object on stdout and diagnostics on stderr:

```json
{
  "schema_version": "falcon/v1",
  "kind": "JobList",
  "data": [],
  "meta": {
    "count": 0,
    "limit": 50
  }
}
```

Conventions:

- keys are stable snake_case;
- quantities include raw numeric resource values in snapshots;
- `null` means unknown/unavailable and `0` means measured zero;
- JSON contains no ANSI escapes or leading/trailing log text;
- lists are bounded by default and report the applied limit in `meta`;
- timestamps use Kubernetes RFC3339 strings or numeric collection times;
- additions may be made within `falcon/v1`; incompatible changes require a new
  schema version.

Kinds currently include `JobDryRun`, `SubmittedJob`, `JobList`, `Job`,
`EventList`, `JobLogs`, `JobMetrics`, `KillResult`, and
`ClusterResources`.

`JobMetrics` reports current/minimum/maximum/arithmetic-average GPU, VRAM, CPU,
and memory utilization over one bounded observation interval. Percentages use
only the current active allocation. The response also includes raw allocation
denominators and the configured average-GPU-utilization eviction floor.
Missing telemetry remains `null`. `falcon metrics` always emits this JSON
object; no output-format flag is required.

`ClusterResources.data.summary` is deliberately compact: node/Job/Pod counts,
capacity, allocatable resources, active requests, request headroom, and GPU
availability by model. Node details are bounded by `--limit`; each node's
visible, non-system consumer list is independently bounded by
`--consumer-limit`. System Pod requests remain included in node and cluster
request totals. Node entries include `gpu_memory_bytes_per_device` when the
cluster publishes the `nvidia.com/gpu.memory` label; `null` means unavailable.

Agents should select on `schema_version` and `kind`, then read `data`.
Human-readable tables are intentionally not a compatibility surface.
