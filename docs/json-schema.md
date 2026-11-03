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
denominators and the configured average GPU/VRAM-utilization eviction floors.
The `eviction_policy` object reports the GPU and VRAM floors and whether each
observed average meets its floor; the dashboard treats either failing metric on
any physical GPU as eviction risk. Monitoring applies a 120-second minimum
startup grace, a 30-second minimum qualification window, and then a rolling
window that grows to five minutes. A slower ten-minute EMA is also exposed for
diagnostics.
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

The MCP transport uses compact Job references: `{"name": "train"}` resolves
the latest run, while an optional `uid` selects a specific run. Namespace is
implicit in Falcon's configuration and omitted from MCP arguments and responses,
including nested events and continuations. Wait responses return the resolved
`jobs` references; a continuation preserves those UIDs. CLI JSON retains its
existing namespace fields.

GPU resource plans can include an additive `scheduling` object. MCP `plan_job`
returns it under `resource_plan.scheduling`, with requested resources, per-node
`available`/`capacity`, `shortfall_now`/`shortfall_capacity`, jointly fitting
alternatives, maximum and recommended allocations, and a `next_action`.
CPU is expressed in cores, memory in GiB, and GPUs in device counts; allocation
suggestions use Kubernetes quantity strings. `gpu_limits.max_per_node` and
`max_free_per_node` are single-node limits, not cluster sums. With no free
matching GPUs, recommendation/maxima basis is `node_capacity`; otherwise it is
`currently_available`. Unknown availability is null. These are request-headroom
snapshots, not measured utilization or reservations. Per-node details are
limited to 20 and alternatives to 3; `matching_node_count` and `nodes_truncated`
identify omitted details. Separate CPU/RAM maxima may come from different nodes.
