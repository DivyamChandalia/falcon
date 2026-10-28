---
name: falcon
description: Manage GPU and CPU Kubernetes Jobs through Falcon's MCP tools.
---

# Falcon

## CPU and memory sizing

- Request only CPU and host memory the task needs; estimate workers/threads, data and batch sizes, and modest headroom, not speculative reserves.
- GPU Jobs: start `plan_job` with GPU model/count only; command/CPU/memory are optional. Leave `maximize` off; use its node-aware recommended allocation as baseline, less when sufficient, and tune batch/concurrency/model settings to maximize GPU compute and VRAM utilization whenever the workload allows.
- Read `scheduling.gpu_limits.max_per_node` as the maximum requestable GPU count on one node and `max_free_per_node` as current headroom; never sum across nodes or exceed the former; re-plan after changing count. Do not hard-code per-GPU CPU/RAM values.
- Use per-node CPU/RAM pairs, not independent maxima. Only when no GPU is free, maxima/recommendations use node capacity, not current CPU/RAM headroom; if required resources fit capacity and no suitable alternative exists, submit unchanged and wait with `wait_for_capacity`.
- Only exceed auto-allocation when absolutely necessary; explain concrete evidence such as measured CPU bottlenecks or confirmed host-memory OOM/peak working-set needs. GPU VRAM OOM is not evidence for more host RAM.
- CPU-only Jobs require explicit CPU/memory: 1 CPU and 1Gi suit trivial checks/smoke tests; size real workloads from task requirements or measured peaks, not that fixed default.

## MCP lifecycle

- Use `plan_job` before uncertain sizing; `submit_job` with a unique `idempotency_key` (reuse on ambiguity). References use `{name}` for the latest run or `{name, uid}` for a specific run; namespace is implicit.
- If supplying Job `name`, use 1–63 lowercase letters/digits/hyphens, starting and ending alphanumeric; replace underscores in the name only, not model arguments or paths.
- `submit_job` only submits: report its Job reference, then start monitoring as a separate explicit call unless asked not to watch. Batch all relevant Jobs into one `wait_for_events` call; one cursor covers them, and separate waits are only needed for different filters or scheduling policies.
- Claude Code 2.1.212+: call `wait_for_events` directly in the main conversation and let Claude Code background the long call; do not create a subagent or poll. Keep Claude and the MCP server open, process the completion notification, and re-arm only after the wait ends, expires, or is disconnected. Older Claude versions block the turn; setup warns and asks for an upgrade.
- OpenCode: one background report-only subagent with its configured subagent model. Give it references/cursor/filters/startup grace/policy. It only awaits synchronously, renews continuations and reports bounded events/cursor/settings; no inspection, edits, restarts, cancellation or polling.
- Codex: start one async `wait_for_events` in `functions.exec`; save its yielded cell ID.
  Make `functions.wait(cell_id=..., yield_time_ms=604800000)` the last action of every turn unless the user explicitly asks to stop watching. This requests a seven-day host wait; respect host caps if rejected.
  Report updates before waiting; do not end with a final response while Jobs are still under watch. Start with `cursor=0`; save every returned cursor.
  If the user interrupts, answer them, then reattach to the same surviving cell with `functions.wait`.
  If the MCP call/cell was actually killed or lost, start one replacement async wait with the same
  references, last cursor (0 until the first result), filters, startup grace and policy; save its new cell ID.
  A yielded or completed cell is not a killed call: retrieve its result; never start duplicate pending waits.
  Keep MCP open: its watcher records intervening events for cursor-based replay; never resubmit Jobs.
  If asked to stop watching, do not re-arm; leave the Job running unless separately asked to cancel it.
- OpenCode parents may end the foreground turn; handle events, then re-arm. Never poll task status, Jobs, logs or metrics, or do busywork while waiting.
- `list_jobs`, `inspect_job`, `job_logs`, `job_metrics`: targeted inspection only; missing usage is unknown.
- Failures: inspect logs, use `restart_job` to relaunch or `cancel_job` to stop.
- `restart_job`: read logs, then pass a short `summary={"issue": "…", "fix": "…"}` describing the failure and your fix. Logs are archived before replacement; same name, new UID: monitor the new reference.

## Scheduling and eviction risk

- "Schedule for later", "queue", "wait for a GPU": submit now, use `scheduling_policy="wait_for_capacity"` on each wait.
  Capacity shortages stay silent; other errors, completion and eviction risk return. Do not resize/duplicate queued Jobs.
- "Run now"/unspecified: `scheduling_policy="report_blocked"` reports blockers; both policies submit now.
- Wait without polling; reuse continuations. Outages retry silently; cancellation leaves the Job running. Events entirely offline may be missed.
- GPU/VRAM floors per physical GPU: H100/PRO6000 use 75%, A6000/2080Ti use 10%.
  Either metric below its floor raises risk after at least 120s startup grace and a 30-second minimum rolling window growing to five minutes.
