# Falcon

### Run GPU jobs on Kubernetes like local commands.

Falcon turns an ordinary command into a well-sized Kubernetes Job, then gives
you a fast terminal dashboard for understanding what is running, waiting,
failing, and consuming the cluster.

<p align="center">
  <img src="assets/falcon-dashboard.svg" alt="Falcon Jobs dashboard" width="100%">
</p>

## Why Falcon

Without Falcon, starting one experiment can mean writing Job YAML, GPU
resources, node selectors, mounts, shared memory, environment plumbing, and
cleanup rules.

With Falcon:

```console
falcon h100x2 -- python train.py
```

Falcon discovers cluster capacity, creates the Job directly, preserves the
current Python environment when requested, and keeps request history correct
after Pods disappear.

## Install

Requires Python 3.10+, `kubectl`, and access to a Kubernetes cluster.

```console
# From a Falcon checkout:
python -m pip install .
falcon setup
```

Setup writes `~/.falconrc` and optional shell completion. `falcon` remains a
normal console executable: no shell function, login shell, or environment
activation is required to call it.

## Tab completion

Falcon completes commands, GPU presets, options, and current Job names:

```text
falcon <TAB>               commands and GPU presets
falcon h<TAB>              h100
falcon h100<TAB>           h100x2, h100x3, …, h100x8
falcon h100x2 --<TAB>      launch options
falcon logs <TAB>          current Kubernetes Job names
falcon resources --<TAB>   resource filters and output options
```

Open a new shell after setup, or load completion immediately:

```console
eval "$(falcon completion zsh)"   # use bash for Bash
```

Completion is cached briefly, so completing Job names does not query
Kubernetes on every keypress.

## Run a Job

For normal GPU work, use a preset directly:

```console
falcon h100 -- python train.py
falcon h100x2 -- python train.py --epochs 100
falcon -c 8 -m 32Gi -- python preprocess.py
```

Falcon turns the command after `--` into a Kubernetes Job, applies resource
requests, mounts the selected environment, creates it directly, and returns.
Add `-f`/`--follow` to stay attached to its logs; Ctrl+C then kills that Job.
Add `--dry-run` to inspect the manifest without creating the Job:

```console
falcon h100x2 --dry-run --output json -- python train.py
falcon -c 8 -m 32Gi --dry-run --output json -- python preprocess.py
```

Omit `-- COMMAND` for a disposable interactive shell:

```console
falcon 2080ti
```

Falcon shows the requested GPU/CPU/RAM while it waits, then opens your current
Zsh/Bash in the same working directory with a `(2080ti)` prompt marker. Your
shell config is loaded without letting Conda startup replace Falcon's selected
environment. The debug Job is deleted when you exit or press Ctrl+C.

Automatic GPU sizing leaves 1 GiB of RAM outside the container request as a
safety buffer. Pass `-m` to override memory explicitly without also overriding
CPU.

Commands are passed as argv. Falcon never wraps them in nested shell strings.

## Monitor

```console
falcon dashboard
```

The Jobs dashboard separates durable requests from current allocations,
aggregates every Pod attempt and container restart, retains last-known-good
data during API failures, and offers keyboard and mouse navigation.

## Understand cluster resources

```console
falcon resources
```

<p align="center">
  <img src="assets/falcon-resources.svg" alt="Falcon cluster resources dashboard" width="100%">
</p>

<p align="center"><sub>Live cluster resource snapshot captured from Falcon.</sub></p>

The resources dashboard reports request headroom—not guessed usage—and lets
you expand a node to answer:

> Why is this node busy, and which namespace or workload is using it?

It shows Falcon Jobs and other meaningful Pods by namespace and workload.
Structured output retains `owner_identity` only when a reliable label or
annotation provides one.

## Coding agents

Agents use the same small command surface as humans:

```console
falcon h100x2 --name experiment -- python train.py
falcon logs experiment --no-follow --tail 100
falcon metrics experiment --interval 60
falcon kill experiment
```

After launching, an agent should report the Job name and return unless you
explicitly ask it to observe the Job. Metrics use current allocations. H100
Jobs require at least 90% average GPU utilization under the eviction policy;
A6000 and 2080Ti Jobs require at least 30%.

Install the single concise Falcon skill:

```console
falcon setup --non-interactive --install-skills codex,claude,opencode
```

The installer follows each agent’s native skill directory, is idempotent, and
will not overwrite a user-modified skill.

## Key features

- Direct `batch/v1` Job manifests and Kubernetes operations—no hidden CLI
  subprocess abstraction.
- Honest requested, allocated, and observed resource semantics.
- Retry history across every Job Pod, including total container restarts.
- Responsive Jobs and cluster-resource TUIs with stale-data retention.
- Bounded human/JSON commands with meaningful exit codes.
- Explicit Conda/venv selection without activation or login-shell wrappers.
- Deterministic demo data, interaction tests, and a 12-dimension visual
  regression matrix.

## Useful commands

```text
falcon h100 -- COMMAND             run a GPU Job
falcon -c CPU -m MEMORY -- COMMAND run a CPU-only Job
falcon jobs [filters]              list bounded Jobs
falcon get JOB                     inspect resources and attempts
falcon events JOB                  inspect bounded Kubernetes events
falcon logs JOB --no-follow --tail 100
falcon metrics JOB --interval 60   return allocation-scoped JSON metrics
falcon kill JOB                    kill a Job
falcon dashboard                   open the Jobs TUI
falcon resources                   open the node/resource TUI
falcon setup                       configure Falcon and agent skills
```

GPU shorthand remains intentionally small:

```console
falcon h100 -- python train.py
falcon h100x2 -- python train.py
falcon a6000 -- python train.py
falcon 2080ti -- python train.py
```

Common one-letter forms are also available: `j` jobs, `g` get, `e` events,
`l` logs, `a` attach, `t` top, `m` metrics, `k` kill, `c` clean, `d`
dashboard, `r` resources, and `s` setup.

## Documentation

- [CLI reference](docs/cli.md)
- [Configuration](docs/configuration.md)
- [Resource semantics](docs/resource-semantics.md)
- [JSON schema](docs/json-schema.md)
- [TUI controls](docs/tui.md)
- [Agent skills](docs/agents.md)
- [Architecture](docs/architecture.md)
- [Development](docs/development.md)
- [Migration notes](docs/migration.md)

## Development

```console
python -m pytest
python -m build
python scripts/capture_tui_snapshots.py
python scripts/capture_live_resources.py
```

The optional kind suite runs only against an explicitly selected disposable
kind cluster:

```console
FALCON_KIND_INTEGRATION=1 python -m unittest tests.test_kind_integration -v
```

Falcon is licensed under Apache-2.0. See [NOTICE](NOTICE) for attribution.
