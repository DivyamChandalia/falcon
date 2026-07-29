# CLI reference

All commands accept `--help`; `falcon --help` is the authoritative option list.

## Launch

```text
falcon PRESET[xN] [options] -- COMMAND...
falcon --gpu MODEL --gpus N [options] -- COMMAND...
falcon -c CPU -m MEMORY [options] -- COMMAND...
```

CPU-only Jobs require `--cpu` and `--memory`. GPU Jobs are sized against
schedulable matching nodes. Automatic GPU memory sizing leaves a 1 GiB safety
buffer. CPU and memory may be overridden independently; an explicit `-m`
replaces the buffered automatic memory value. `--environment auto` honors
`VIRTUAL_ENV` before `CONDA_PREFIX`;
`--environment none` disables environment mounting; a path selects explicitly.

`--dry-run` performs no create operation. With `--output json`, it emits the
request, resource plan, and manifest as structured data.

Launches are detached by default. `-f`/`--follow` stays attached to the new
Job's logs; pressing Ctrl+C in that mode kills the Job Falcon just created.
`--follow` cannot be combined with JSON launch output.

When `COMMAND` is omitted, Falcon creates a bounded debug Job, waits for its
Pod, opens the invoking shell (`zsh` or `bash`) in the launch directory, and
prefixes its prompt with the preset, such as `(2080ti)` or `(2080tix2)`.
The waiting line shows the exact requested GPU, CPU, RAM, and shared memory.
Falcon sources the detected `.zshrc` or `.bashrc`, while restoring the selected
Conda/venv identity afterward so shell startup cannot replace the Job
environment. Falcon always deletes the Job when the shell exits or Falcon is
interrupted. Commandless debug sessions use zero retries and a six-hour safety
deadline.

## Inspect

```text
falcon jobs [--limit 50] [--status STATUS] [--gpu MODEL] [--node NODE]
falcon get JOB
falcon events JOB [--limit 50] [-f|--follow]
falcon logs [JOB] [--tail 100] [--follow|--no-follow] [--output human|json]
falcon metrics JOB [--interval 10]
falcon top [JOB]
```

`jobs`, `get`, `events`, and logs support `--output human|json`. Human `logs`
follows and stays attached by default;
`--no-follow` prints one bounded tail. JSON logs are always bounded and exit.
`events -f` polls for new or updated Job/Pod events until interrupted.
`get` deliberately excludes events; use the dedicated `events` command.
`metrics` always returns JSON, observes one Job for the requested duration, and returns
current/minimum/maximum/average GPU, VRAM, CPU, and memory utilization. Every
percentage uses the Job's current active allocation as its denominator.
`top` opens interactive nvitop. Omitting `JOB` for logs/attach/top uses Falcon’s
last successful launch or inspected target when available.

## Mutate

```text
falcon kill JOB [JOB...]
falcon clean
```

## TUI

```text
falcon dashboard [--job JOB] [--demo [STATE]]
falcon resources [--node TEXT] [--gpu TEXT] [--limit 100] [--demo [STATE]]
```

Resources accounts for active requests from every namespace so node totals
reconcile, while hiding infrastructure Pods from consumer rows and Pod counts.
Pass `--namespace` only for a deliberately scoped view. JSON returns at most
100 visible workloads per node by default; raise that bound with
`--consumer-limit` when inspecting an unusually busy node.

Non-TTY processes should use the domain commands with `--output json` rather
than attempting to scrape either dashboard.

## Setup

```text
falcon setup [--non-interactive] [--no-shell]
falcon setup --install-skills codex,claude,opencode
falcon setup --uninstall-skills codex
falcon completion [bash|zsh]
falcon config
```

Exit codes are `0` success, `2` input/config error, `3` Kubernetes unavailable
or operation failure, `4` object not found, `5` safe-install conflict, and
`130` interrupted.

## Short aliases

`j` jobs, `g` get, `e` events, `l` logs, `a` attach, `t` top, `m` metrics,
`k` kill, `c` clean, `d` dashboard, `r` resources, and `s` setup. Completion
shows base GPU presets first; after `falcon h100`, `falcon a6000`, or
`falcon 2080ti`, the next Tab completes same-token forms such as
`2080tix2`. One-letter aliases remain accepted but are intentionally omitted
from completion suggestions.
