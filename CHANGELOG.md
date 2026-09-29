# Changelog

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
