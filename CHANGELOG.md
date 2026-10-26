# Changelog

## 0.5.0 — 2026-10-26

This release consolidates the agent-managed Job lifecycle, event-driven
Dashboard updates, and MCP setup work merged since `0.4.7`. It is the first
release where Falcon's MCP support is part of the normal installation and is
intended to be used as the primary interface for coding agents that manage
long-running Jobs.

### MCP setup and agent workflow

- Falcon now includes the MCP SDK in the normal installation. The MCP server
  runs as a local stdio process, uses the user's existing kubeconfig and
  configured namespace, and does not reserve cluster resources or grant extra
  Kubernetes permissions.
- `falcon setup` can configure Falcon MCP for Codex, Claude Code, and OpenCode,
  with safe config merges, timestamped backups, dry-run support, and a prompt
  before client files are changed. The MCP-first skill is installed by default;
  `--skill-mode legacy` remains available for users who intentionally want the
  direct-CLI workflow.
- After upgrading to `0.5.0`, run `falcon setup` again so Falcon can install or
  refresh the MCP entry and the managed agent skill. Setup preserves edited or
  unmanaged files and records whether each existing agent uses MCP or legacy
  mode. `falcon update` refreshes already-managed skills without opting new
  agents in unexpectedly.
- Added `falcon mcp config` snippets for all supported clients and retained
  configurable MCP execution windows for long-running Jobs. Claude Code setup
  checks whether the installed client can background long MCP calls and warns
  when an upgrade is needed.
- Agent guidance now recommends task-sized CPU and memory requests, using
  Falcon's node-aware GPU allocation as a baseline instead of requesting
  speculative resources. CPU-only smoke tests default to a small 1 CPU/1Gi
  footprint, while larger requests require a concrete workload reason.

### Planning, submission, and restart lifecycle

- Added MCP tools to plan, submit, inspect, list, measure, follow, cancel, and
  restart Jobs. `plan_job` can be resource-only and returns node-specific
  availability, same-node CPU/RAM pairs, exact shortfalls, maximum requestable
  GPU counts, recommendations, and alternatives before a Job is created.
- `submit_job` is deliberately separate from monitoring: it submits once,
  returns a compact `{name, uid}` reference, and does not silently start a
  watcher. Agents explicitly call `wait_for_events` when the user wants the
  Job followed. Namespace is taken from Falcon configuration rather than
  repeated in every tool argument or response.
- MCP-created Jobs use idempotency keys and disable automatic Kubernetes
  retries, returning failures to the agent for an explicit decision. Generated
  names are readable `falcon-<word>-<word>-<number>` names. Reusing a name with
  a different Falcon key remains safely rejected, but the error now explains
  whether the name belongs to another Falcon submission or has no Falcon
  marker, without revealing keys or hashes.
- Restarts keep the logical Job name, delete the old Job/Pods safely, and
  create a replacement with a new UID. Restart identity is tracked internally,
  so interrupted restart calls can resume without another agent-generated key.
  UID and resource-version guards prevent stale warnings or unrelated
  finalizers from affecting a replacement.
- Restarted attempts archive their old Pod logs locally and accept a short
  issue/fix summary. Dashboard users can switch between attempts, inspect
  failed-only logs, and select/copy the summary below the failed Pod logs.
  Cancelled Jobs are still deleted normally and are not treated as archived
  restart attempts.

### Long-running monitoring and scheduling policy

- `wait_for_events` is a server-side Kubernetes watch rather than agent-side
  polling. It retains a cursor and records events while the agent answers a
  user. Codex uses one async host wait that can be reattached with
  `functions.wait`; only a killed or lost call is replaced. Claude Code and
  OpenCode use their supported background/report-only workflows.
- The agent-facing wait does not expose a short timeout that encourages
  polling. The packaged Codex workflow requests a seven-day host wait and can
  continue from the returned cursor for another seven days. Monitoring can be
  stopped without cancelling the workload.
- Added `scheduling_policy="wait_for_capacity"` for “schedule/queue and follow”
  requests. Recognized resource shortages remain quiet while Kubernetes
  schedules the Job; completion, failures, placement errors, image/mount
  failures, and eviction risk still wake the agent. The default
  `report_blocked` policy reports scheduling blockers for “run now” requests.
- GPU eviction risk is evaluated per physical GPU using GPU and VRAM floors,
  with a 120-second minimum startup grace and a rolling window that begins at
  30 seconds and grows to five minutes. Sampling gaps and irregular intervals
  no longer leave a risk permanently stuck in qualification.
- Monitor health transitions are kept internal, outages retry until recovery,
  persistent failures are debounced, and re-armed waits avoid repeating
  namespace-wide baseline lists. Fixed watcher recovery on kubectl versions
  that do not support `get --resource-version` by resuming through the raw
  Kubernetes API with the resource version in the query.

### Dashboard and resource observability

- Dashboard Job, Pod, and Event inventory is now event-driven through
  Kubernetes watches, reducing repeated full inventory downloads while keeping
  metrics sampling at its existing cadence. Resource and Dashboard views share
  the live inventory projection and recover cleanly from expired resource
  versions or temporary API interruptions.
- Hardened Dashboard and MCP shutdown paths so watch/list workers are cancelled
  and closed cleanly instead of leaving stale executor state in a long-lived
  client process or a subsequent terminal session.
- Dashboard agent-managed Jobs retain their normal status colours. While an
  active `wait_for_events` connection is monitoring a Job, its status cell
  displays the animated `◴◷◶◵` symbol. This symbol means “Falcon is actively
  monitoring this Job”; it does not indicate remaining model quota or a new
  workload state. Pending and eviction-risk colours remain unchanged. When the
  wait disconnects, the animation disappears and the ordinary status dot
  returns. Failed Jobs retain `✕`, and succeeded Jobs retain `✓`.
- Added local, expiring agent-presence state scoped by Kubernetes context,
  namespace, and Job UID. Presence expires after a crashed MCP process and is
  cleared immediately on normal wait cancellation or return, so a stale agent
  indicator does not persist indefinitely.
- Preserved the existing Resources visual language while adding combined CPU,
  memory, and GPU allocation views, system-wide namespace charts, shared
  sorting, compact Job ages, namespace/workload filtering, and individually
  selectable expanded panels.
- Dashboard logs and terminal panes support text search, highlighted matches,
  mouse selection, edge auto-scroll, clipboard copying, selectable restart
  summaries, and atomic tqdm carriage-return updates that avoid delete/repaint
  flicker.

### Compatibility and safety

- Existing CLI launches, retry settings, Coder-owned Jobs, mounts, and
  Kubernetes permissions remain unchanged. MCP operations use the configured
  namespace and compact Job references, while exact UIDs remain available for
  historical attempts and archived logs.
- Added broad unit coverage for MCP service validation, event cursors,
  scheduling policies, watch recovery, GPU/VRAM risk, restart archives,
  dashboard inventory, agent presence, client setup, skill refreshes, and
  snapshot behavior.

## 0.4.7 — 2026-10-13

- Added Ctrl+F search in Dashboard logs and terminals, with highlighted
  matches, next/previous navigation, and isolated search input.
- Added quoted wildcard patterns and `--dry-run` to `falcon kill`, with
  namespace-scoped matching, deduplication, and preserved Coder handling.
- Added compact Job ages to Resources tables and fixed GPU namespace pies
  switching between GPU count and VRAM with `v`.

## 0.4.6 — 2026-10-01

- Added interactive Dashboard terminals: Ctrl+T opens shells, arrows switch
  windows, Ctrl+D closes, and 200-line scrollback supports selection-aware
  copying. Click outside the terminal to restore Dashboard controls.
- Added Resources namespace/workload filters with `f`, single-pane focus,
  contextual filter hints, and cleaner combined/expanded views without
  redundant Nodes/Allocations selectors.
- Added uv-aware `falcon update` and uv installation instructions; Dashboard
  `k` now opens Job actions from any non-terminal pane.

## 0.4.5 — 2026-09-29

- Added mouse text selection, edge auto-scroll, and clipboard copy for
  Dashboard logs.
- Replaced tqdm progress rows atomically to remove the delete-and-repaint
  flicker during carriage-return updates.
- Made `falcon update` show up to three concise “What’s new” entries after a
  successful upgrade.

## 0.4.4 — 2026-09-28

- Added the combined Resources view with shared sorting and CPU, memory, and
  GPU allocation visualizations.
- Added expanded system-wide namespace charts and refreshed the Resources
  documentation and captures.

## 0.4.3 — 2026-09-22

- Added an interactive `falcon setup` confirmation before copying an external
  `kubectl` into `$HOME/.local/bin`; pressing Enter accepts the default Yes.
- Documented that the shared-home copy makes `kubectl` and Falcon available in
  Coder sessions for scheduling Jobs, while retaining the kubeconfig and RBAC
  prerequisites.

## 0.4.2 — 2026-09-18

- Added tqdm-compatible log capture: carriage-return progress updates replace
  the active bar, newline-normalized updates collapse to the latest bar, and
  ANSI terminal controls are removed from the rendered text.
- Preserved natural wrapping for bars that genuinely exceed the available
  widget width while keeping ordinary log text behavior unchanged.

## 0.4.1 — 2026-09-18

- Made Dashboard log scrolling smoother with a virtualized, incrementally
  updated log viewport that retains line wrapping and auto-follow behavior.
- Removed the log widget's gray surface by restoring Falcon's black background
  and white foreground in both normal and focused states.
- Coalesced rapid Jobs scrolling and selection changes into one render per
  terminal frame, avoiding unnecessary full-dashboard rebuilds.

## 0.4.0 — 2026-09-18

- Added the responsive Resources layout, with Nodes and Selected Node on the
  left and GPU Allocations on the right at `160×30` and above.
- Added shared Namespace, CPU, memory, and GPU sorting for GPU-requesting Jobs
  and Selected Node workloads, including expanded panes.
- Added GPU, requested-memory, and requested-CPU allocation modes and the
  aspect-correct GPU-by-namespace chart layout, including a toggleable
  logarithmic Allocation History scale.
- Kept GPU request and mark columns visible in narrow Dashboard tables and
  refreshed deterministic TUI captures and documentation.

## 0.3.0 — 2026-09-17

- Added `falcon update` and `falcon update --check` for GitHub-based upgrades.
- Added a once-per-24-hour interactive update prompt for human-facing TTY
  commands, with JSON/non-interactive safety and `FALCON_NO_UPDATE_CHECK=1`
  opt-out support.
- Centralized the package version in `falcon/__init__.py` and made setuptools
  derive wheel metadata from it.
- Added update/versioning documentation, shell completion, and test coverage.
- Fixed expanded Selected Job metadata overlap that caused Eviction risk text
  to jitter when the pane was clicked.
