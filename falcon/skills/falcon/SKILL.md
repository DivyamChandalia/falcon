---
name: falcon
description: Manage GPU and CPU Kubernetes Jobs through Falcon's MCP tools.
---

# Falcon

Use Falcon's MCP tools first. Configure with `falcon setup` or
`falcon setup --non-interactive --configure-mcp all`; the SDK is included.
Use CLI only as fallback; never use an interactive shell or `conda activate`.

## MCP lifecycle

- Call `plan_job` before uncertain submissions, then `submit_job` with a unique
  `idempotency_key`; reuse that key after an ambiguous response.
- Immediately report submitted Jobs and say you are awaiting events. Start one
  `wait_for_events` with returned references; batch Jobs and reuse the cursor.
  Await it synchronously: do not background, detach, or end the turn while it
  is pending. Keep the session open; do not poll or do busywork while waiting.
- Use `list_jobs`, `inspect_job`, `job_logs`, and `job_metrics` for targeted
  inspection only, never timer polling. Missing measured usage is unknown.
- On failure, inspect pod logs: prefer `restart_job` for repairable terminal Jobs
  or `cancel_job` to stop. Falcon never retries automatically.
- `restart_job` keeps the name but returns a new UID; monitor that reference.
  Old Pods/logs remain readable with the old reference. Use `submit_job` only
  for independent runs or unsupported overrides; cancel active Jobs first.
- Use `cancel_job` only with the exact returned Job reference and UID.

## Monitoring and safety

- "Schedule for later", "queue", "wait for a GPU": submit now and wait with
  `scheduling_policy="wait_for_capacity"`; reuse the policy with each cursor.
  Capacity shortages stay silent; placement, image/mount errors, failure,
  completion, and eviction risk still return. Do not resize/duplicate queued Jobs.
- "Run now" or unspecified: default `scheduling_policy="report_blocked"` reports
  scheduling blockers. Both submit now; neither reserves a future start time.
- `wait_for_events` waits up to seven days; on expiry, reuse its continuation.
  Outages retry silently until recovery; cancellation leaves the Job running.
- GPU/VRAM floors per physical GPU: H100/PRO6000 use 75%, A6000/2080Ti use 10%.
  Either metric below its floor raises risk after at least 120s startup grace
  and a 30-second minimum rolling window growing to five minutes.
- MCP Jobs use `backoffLimit=0` and `restartPolicy=Never`; a cancelled wait
  does not cancel the Job. Deleted Jobs disconnect silently.
- Coder-owned Jobs remain managed by Coder. MCP changes no client, host, or
  Kubernetes permissions. If MCP is unavailable, use the normal Falcon CLI.
