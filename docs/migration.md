# Migration to Falcon 0.2

User-visible changes:

- The distribution is `falcon-k8s` and installs only `falcon`.
- `falcon h100[xN] -- COMMAND` is the GPU launch form.
- `falcon -c CPU -m MEMORY -- COMMAND` is the CPU-only launch form.
- The redundant `falcon submit` command was removed; its error names both
  direct replacements.
- `falcon kill` replaces `falcon delete`.
- Bounded utilization moved from `top --samples` to
  `falcon metrics JOB --interval SECONDS`; `top` is interactive only.
- Launches now return immediately by default. Add `-f`/`--follow` only for a
  foreground launch whose Job should be killed when Falcon is interrupted.
- A launch with no `-- COMMAND` remains an interactive debug shell and is
  deleted automatically on shell exit or Ctrl+C.
- `metrics` emits JSON by default, `get` no longer embeds events, and
  `events -f` follows new events.
- Operational commands have one-letter aliases (`j`, `g`, `e`, `l`, `a`, `t`,
  `m`, `k`, `c`, `d`, `r`, `s`).
- Raw launcher arguments and generated-command abstractions were removed.
- Agent data moved from dashboard snapshots to `jobs`, `get`, `events`, and
  `resources` with `--output json`.
- Namespace can be overridden on noninteractive commands with `--namespace`.
- `falcon resources` is now a realtime node/resource TUI in a terminal and a
  bounded table/JSON response when piped.
- Setup installs completion only; it no longer creates a shell function or
  pins a launcher to one Python environment. Running setup also removes the
  obsolete `falcon shell-init` line installed by preview releases.
- A CPU-only submission must explicitly provide both CPU and RAM.

The old dashboard JSON flags fail with a message naming the replacement.
Convenience `logs`, `attach`, `top`, `kill`, and `clean` workflows remain.
