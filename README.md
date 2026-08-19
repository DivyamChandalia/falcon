# Falcon

Run Kubernetes jobs and Coder workspaces from your terminal.

Falcon handles GPU selection, CPU and memory sizing, manifests, logs, cleanup,
and cluster monitoring so you can focus on the command you want to run.

```console
falcon h100x2 -- python train.py
```

<p align="center">
  <img src="assets/falcon-dashboard.svg" alt="Falcon Jobs dashboard" width="100%">
</p>

## Quick start

Falcon requires Python 3.10+, `kubectl`, and access to a Kubernetes cluster.

```console
python -m pip install .
falcon setup
```

`falcon setup` creates `~/.falconrc` and installs shell completion. Open a new
shell after setup, then launch a workload:

```console
# One GPU
falcon h100 -- python train.py

# Multiple GPUs
falcon h100x2 -- python train.py --epochs 100

# CPU-only job
falcon -c 8 -m 32Gi -- python preprocess.py
```

## Run workloads

Falcon includes these GPU presets and limits:

| Preset | Maximum GPUs |
| --- | ---: |
| `h100` | 8 |
| `2080ti` | 4 |
| `a6000` | 2 |
| `pro6000` | 2 |

Append `xN` to request multiple GPUs, such as `2080tix4` or `pro6000x2`.

Useful launch options:

```console
# Follow logs after submitting
falcon h100 -f -- python train.py

# Preview the Kubernetes manifest without creating a Job
falcon pro6000x2 --dry-run --output json -- python train.py

# Override automatic CPU and memory sizing
falcon a6000 -c 12 -m 64Gi -- python train.py

# Open a temporary interactive shell
falcon 2080ti
```

The interactive shell opens in your current working directory and is removed
when you exit.

## Monitor jobs

Open the interactive Jobs dashboard:

```console
falcon dashboard
```

Or use individual commands:

| Command | Purpose |
| --- | --- |
| `falcon jobs` | List jobs |
| `falcon get JOB` | Inspect a job and its attempts |
| `falcon logs JOB` | Follow logs |
| `falcon events JOB` | Show Kubernetes events |
| `falcon metrics JOB` | Show available allocation metrics |
| `falcon kill JOB` | Remove a job |

## Inspect cluster resources

```console
falcon resources
```

<p align="center">
  <img src="assets/falcon-resources.svg" alt="Falcon cluster resources dashboard" width="100%">
</p>

Resources has two views:

- **Nodes** shows free CPU, memory, and GPUs for every node. Select a node and
  press <kbd>Enter</kbd> to inspect the jobs using it.
- **GPU Allocations** shows allocation history, allocation by namespace, and
  active GPU jobs. Press <kbd>v</kbd> to switch between GPU count and VRAM.

Use <kbd>←</kbd>/<kbd>→</kbd> to switch views, <kbd>Tab</kbd> to move focus,
and <kbd>Enter</kbd> to expand the selected pane. Falcon remembers your last
view and keeps GPU allocation history in the background.

<p align="center">
  <img src="assets/falcon-resources-allocations.svg" alt="Falcon GPU allocation history" width="100%">
</p>

Resource values are based on Kubernetes requests and allocations. Falcon does
not present them as measured GPU compute utilization.

## Create a Coder workspace

Create a workspace with CPU and memory:

```console
falcon coder -c 4:4 -m 8Gi:8Gi
```

Or size it from a GPU preset and choose a name:

```console
falcon coder pro6000 -j research
```

Falcon waits for the workspace and prints available links for VS Code,
Antigravity, Cursor, JupyterLab, and the web terminal. Editor links open the
directory where you ran the command.

Print links for an existing workspace:

```console
falcon coder research
```

On your first run, Falcon shows a Coder sign-in link and saves the pasted
session token in Coder's standard session file. Delete a workspace through
Coder by killing its full Kubernetes job name:

```console
falcon kill coder-user-research
```

## Shell completion

Falcon completes presets, valid GPU counts, options, jobs, and Coder workspace
names. To load completion immediately after setup:

```console
eval "$(falcon completion zsh)"  # use bash for Bash
```

Examples:

```text
falcon h100<TAB>          h100x2 … h100x8
falcon pro6000<TAB>       pro6000x2
falcon logs <TAB>         current jobs
falcon coder <TAB>        current Coder workspaces
```

## Configuration

Falcon reads `~/.falconrc`. Common settings include the Kubernetes namespace,
runtime image, GPU label, resource history, presets, and Coder template.

```yaml
cluster:
  namespace: research
  gpu_label: gpu-type
presets:
  pro6000:
    gpu_type: pro6000
    max_count: 2
```

See [Configuration](docs/configuration.md) for all available settings.

## Documentation

- [CLI reference](docs/cli.md)
- [Configuration](docs/configuration.md)
- [TUI controls](docs/tui.md)
- [Resource semantics](docs/resource-semantics.md)
- [JSON schema](docs/json-schema.md)
- [Coder and agent workflows](docs/agents.md)
- [Development](docs/development.md)

Common short aliases are also available: `j` jobs, `g` get, `e` events, `l`
logs, `a` attach, `t` top, `m` metrics, `k` kill, `c` clean, `d` dashboard,
`r` resources, and `s` setup.

Falcon is licensed under Apache-2.0. See [NOTICE](NOTICE) for attribution.
