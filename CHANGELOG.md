# Changelog

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
