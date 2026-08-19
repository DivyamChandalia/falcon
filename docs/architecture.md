# Architecture

Falcon is an independent package. It directly models and creates Kubernetes
`batch/v1` Jobs; it does not bundle or execute another job launcher.

The previous embedded toolkit was removed because Falcon consumed only a
resource parser, a Pod waiter, and a subprocess submission path. Replacing
those narrow pieces produced a smaller boundary than retaining an external
runtime dependency. Apache-2.0 attribution for derived ideas/code remains in
`NOTICE`.

The submission flow is:

```text
CLI argv
  -> JobRequest
  -> ResourcePlan
  -> JobSpecification / Kubernetes manifest
  -> KubernetesClient
  -> SubmittedJob
```

Major components:

- `models.py`: typed requests, plans, environments, and submission results.
- `quantities.py` and `planning.py`: pure parsing and cluster-aware planning.
- `manifest.py`: side-effect-free Job manifest generation.
- `kubernetes.py`: bounded, argv-only Kubernetes operations.
- `cluster.py`: shared inventory, request/allocation semantics, retries, nodes,
  consumers, and last-known-good collection.
- `resources.py`: kube-state-metrics adapter and planning-facing resource
  discovery. It reconstructs the cluster view without requiring Node-list
  RBAC, then emits the same `ClusterSnapshot` used by API collection.
- `dashboard.py` / `dashboard_ui.py`: Job telemetry and the Jobs TUI.
- `resources_ui.py`: cluster/node TUI using the same cluster models.
- `resources_history.py`: detached, single-instance allocation collection and
  concurrent SQLite history storage for the Resources charts.
- `coder.py`: authenticated Coder workspace creation and lifecycle actions, rich-parameter mapping,
  readiness polling, and application-link construction.
- `output.py`: the `falcon/v1` machine-output envelope.
- `agent_skills.py`: safe skill detection, installation, update, and removal.

Collectors poll expensive inventory on a cadence and retain the last valid
snapshot on transient failure. Rendering never replaces missing telemetry with
zero. TUI work runs off the render path and cleanup closes collectors.

The configured kube-state-metrics endpoint is the primary cluster-resource
source. Direct Kubernetes inventory is a fallback for sites that grant
cluster-wide Node and Pod reads. This preserves the proven request-accounting
behavior of the earlier resource command without retaining its package or CLI.

The deterministic fixtures in `demo.py` are shared by tests, `--demo`, visual
captures, and the dashboard README asset. Both Resources README assets are
captured from the current cluster by `scripts/capture_live_resources.py`.
Both paths render the real Falcon UI rather than maintaining mock artwork.
