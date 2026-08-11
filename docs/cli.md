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

Before a Job is created, human launches print a compact resolved request
summary (namespace, image, command, and CPU/RAM/GPU/shared-memory values).
With `--output json`, the same summary is sent to stderr so stdout remains
valid JSON.

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

`falcon kill` recognizes Coder-owned Job names in the form
`coder-<current-user>-<workspace>`. It submits Coder's native workspace delete
transition instead of deleting the Kubernetes Job underneath Coder, so the
workspace record and template resources are cleaned up together. This also
works when that Job has already disappeared. Other Job names retain the normal
Kubernetes deletion path, including when both kinds are supplied together.
The dashboard applies the same ownership rule: deleting a Coder Job deletes
the workspace through Coder, while Restart uses Coder's native stop/start
lifecycle and waits for its agent to reconnect. The dashboard intentionally
offers no pod-only deletion action; its guarded operations are whole-Job
Delete and Restart.

## Coder workspaces

```text
falcon coder [GPU_PRESET] [-c CPU[:LIMIT]] [-m MEMORY[:LIMIT]]
             [--template NAME] [--name NAME]
             [--access all|terminal|vscode|cursor|jupyter|APP]
             [--parameter NAME=VALUE] [--timeout SECONDS]
falcon coder WORKSPACE_OR_JOB [--access APP]
```

Coder owns the Kubernetes Job and agent credential; Falcon calls its workspace
API, maps normalized CPU/RAM values to the template's rich parameters, waits
for the agent, and prints OSC 8 editor and terminal links. Generated workspace
names use `color-animal-number`, matching the Coder dashboard. VS Code,
Antigravity, and Cursor links open the directory from which `falcon coder` was
invoked. Use
`--access terminal` when only the web terminal is wanted. Extra template
parameters are repeatable. Antigravity 2.0 is printed only as an
`Antigravity 2.0 IDE` link using the `antigravity-ide` URL scheme. When an interactive run has no
credential, Falcon prints a clickable `/cli-auth` link, reads the pasted token
without echoing it, validates it, and saves it as the standard Coder CLI
session. Non-interactive runs use `CODER_SESSION_TOKEN` or an existing Coder
CLI login and never prompt.
Template-managed JupyterLab apps are included through Coder's authenticated
workspace-app proxy; `--access jupyter` and `--access notebook` select that link.

GPU presets use the same cluster-aware proportional CPU/RAM sizing as ordinary
Falcon Jobs and populate the template's GPU model and count parameters. For
example, `falcon coder 2080ti -j falcon` creates workspace `falcon`; Coder names
its Kubernetes Job `coder-<user>-falcon`.

Passing a non-preset positional value reconnects without creating anything.
Both `falcon coder falcon` and `falcon coder coder-<user>-falcon` wait for the
existing agent and print fresh links for the invocation directory.

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
or operation failure, `4` object not found, `5` safe-install conflict, `6`
Coder API/configuration failure, and `130` interrupted.

## Short aliases

`j` jobs, `g` get, `e` events, `l` logs, `a` attach, `t` top, `m` metrics,
`k` kill, `c` clean, `d` dashboard, `r` resources, and `s` setup. Completion
shows base GPU presets first; after `falcon h100`, `falcon a6000`, or
`falcon 2080ti`, the next Tab completes same-token forms such as
`2080tix2`. `falcon coder <Tab>` lists only short workspace names read from
live Coder Job labels—for example, `falcon` or `lime-gull-30`—and no presets or
options. One-letter aliases remain accepted but are intentionally omitted from
completion suggestions.
