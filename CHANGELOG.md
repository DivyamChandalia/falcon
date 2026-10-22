# Changelog

## Unreleased

- MCP restarts now keep the Job name while returning a new UID and creating
  new Pods. Old terminated Pods remain available for logs, and interrupted
  restarts can be safely resumed with the same idempotency key.

- Simplified agent monitoring to one synchronous, long-lived MCP wait; removed
  advice to background the call and end the turn before an event arrives.

- Fixed MCP GPU/VRAM risk remaining stuck in qualification when real sampling
  intervals do not align exactly with the minimum rolling window.

- Agent-facing `wait_for_events` now always waits up to seven days, without
  exposing a timeout parameter that encourages short polling calls. On expiry,
  a continuation lets the agent follow the same Jobs for another seven days.

- Dashboard agent animation now requires an active `wait_for_events`
  connection; otherwise Jobs retain their original status dot and colour.
  Retained background watches alone no longer imply agent presence.
  Connected Pending and eviction-risk Jobs also animate, keeping their colours.
- Fixed wait reconnection indicators remaining stale while Dashboard metrics
  requests are blocked or Kubernetes inventory is unchanged.

- Added `wait_for_events(scheduling_policy="wait_for_capacity")` for agents
  scheduling Jobs that may queue for resources; other errors still wake the
  agent. Restored Pending status colours while retaining the managed-running
  animation.

- Dashboard Job, Pod, and Event updates now use Kubernetes watches for prompt
  status changes and fewer full inventory downloads; metrics retain their
  existing sampling intervals.
- Fixed MCP monitoring on kubectl versions without `get --resource-version`:
  watches now resume through raw Kubernetes API URLs with the resource version
  preserved in the query, avoiding flag errors and needless reconnect gaps.

- Added optional MCP job tools for coding agents, with server-side event waits,
  bounded per-attempt logs, metrics, and retry-safe submission/relaunch.
- Agent-managed Jobs disable automatic retries so failures return control to
  the agent. Existing CLI retry settings and agent permissions are unchanged.
- Added `falcon mcp config` snippets for Codex, Claude Code, and OpenCode.
- `falcon setup` can now configure Falcon MCP automatically for Codex, Claude
  Code, and OpenCode, with safe merges, backups, and a dry-run option.
- Added `--skill-mode legacy` to retain the pre-MCP direct-CLI agent skill.
- `falcon update` now refreshes existing Falcon-managed agent skills while
  preserving each agent's MCP or legacy mode and protecting edited copies.
- MCP event waits now keep all monitor health changes internal, retry outages
  until recovery without an outage deadline (subject to wait timeout/cancellation),
  debounce persistent outages, and avoid repeating namespace-wide baseline
  lists when an active wait is re-armed.

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
