---
name: falcon
description: Launch, observe, log, and kill Kubernetes Jobs with Falcon. Use for GPU or CPU Jobs requested by the user.
---

# Falcon

Prefer Falcon's MCP tools for agent-managed long-running Jobs (configure with `falcon mcp config --agent codex|claude|opencode`); use `falcon` directly for normal CLI Jobs. Use the same commands a human uses; never wrap Falcon in an interactive shell or `conda activate`.

## Commands

- `falcon h100x2 --name experiment -- python train.py`
- `falcon -c 8 -m 32Gi -- python preprocess.py`
- `falcon logs JOB --no-follow --tail 100`
- `falcon metrics JOB --interval 60`
- `falcon kill JOB`

## Behavior

- Launch the Job, report its name, and return. Do not add `-f`, wait, poll, fetch logs, or observe utilization unless the user explicitly asks to follow or verify it.
- If the user asks to observe the Job or ensure utilization is sufficient, use `metrics`; its GPU, VRAM, CPU, and memory percentages are based on current allocation.
- Eviction policy uses 60-second average GPU and VRAM utilization: H100 and PRO6000 must stay at or above 75%; A6000 and 2080Ti must stay at or above 10%.
- Falcon automatically leaves a 1 GiB memory safety buffer. Use `-m` only when the user explicitly requests a memory value or the workload demonstrably needs it.
- For status, events, manifests, scheduling details, or anything outside launch/log/metrics/kill, use one targeted `kubectl` command.
- Coder-owned names (`coder-<user>-<workspace>`) are lifecycle-managed by Coder: `falcon kill`, `falcon clean`, and dashboard Delete remove the workspace through Coder, while dashboard Restart uses Coder's stop/start lifecycle. Never delete their Jobs directly with `kubectl`.

## MCP

MCP-created or relaunched Jobs use `backoffLimit=0` and `restartPolicy=Never`, have no automatic recovery, a cancelled wait does not cancel the Job, and idle-agent wake-up is not guaranteed.
MCP changes no client or host permissions; submit/restart require idempotency keys and returned wait cursors must be reused. `restart_job` keeps the Job name but returns a new UID; monitor the returned reference. Old terminated Pods and their logs remain readable using the previous Job reference.
