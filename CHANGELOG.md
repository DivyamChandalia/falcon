# Changelog

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
