# TODO

## Cluster-wide utilization (DCGM + node metrics)

Blocked on: per-pod CPU/memory for other namespaces. Do not build until that
is available.

What is reachable today (plain HTTP, no auth, from cluster hosts):

- DCGM exporter: `http://<node>:9400/metrics` on every GPU node (not node1).
  Per-GPU series labelled `hostname`, `modelName`, `UUID`, and `namespace`/`pod`/
  `container` when a pod holds the GPU. 17–60 KB, ~6–25 ms.
  Useful fields: `DCGM_FI_PROF_GR_ENGINE_ACTIVE` (headline busy %, smoother
  than `DCGM_FI_DEV_GPU_UTIL`), `DCGM_FI_PROF_PIPE_TENSOR_ACTIVE`,
  `DCGM_FI_PROF_DRAM_ACTIVE`, `DCGM_FI_DEV_FB_USED`/`FB_FREE`, temp, power.
- node-exporter: `http://<node>:9100/metrics` on every node. Node-level only
  (no per-process/user/pod labels). Full scrape is 250–820 KB and up to 1.6 s on
  nodex1; filter with `?collect[]=loadavg&collect[]=meminfo&collect[]=pressure`
  (~19 KB) or `?collect[]=cpu` (~214 KB, per-core counters, needs two samples).
- metrics-server (`kubectl top pods`): per-pod CPU/RAM, own namespace only.

Blocked / missing:

- Per-pod CPU/RAM for other namespaces: `pods.metrics.k8s.io` list is forbidden
  cluster-wide. Ask the admin for a ClusterRole granting
  `get,list` on `pods.metrics.k8s.io` (preferred). Alternatives: `nodes/proxy`
  for cadvisor (broader), or deploying a process-exporter.

Planned once unblocked:

- Shared collector fetching DCGM + node-exporter + pod metrics in parallel with
  short timeouts; node URL template configurable (`http://{node}:9400/metrics`),
  since hostnames may not resolve on every MCP host.
- Dashboard: per-node and per-pod GPU busy %, tensor activity, VRAM, CPU, RAM;
  replace `kubectl exec nvidia-smi` sampling with DCGM.
- `plan_job` / `gpu:"auto"`: add real utilization next to allocation; flag
  idle-but-allocated GPUs (0% busy with VRAM held); compare against preset
  `minimum_utilization`.
- `job_metrics`: GPU numbers from DCGM `pod` label instead of exec.
- Decide privacy: show other namespaces' pods individually or aggregated.
