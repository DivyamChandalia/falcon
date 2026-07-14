# Native Falcon

Falcon provides cluster-aware GPU submission and monitoring on top of Jet.

## Install and configure

```bash
pip install -e .
command falcon setup --force  # use without --force on a fresh install
```

Use `command falcon setup` for the one-time migration because `command` bypasses an older `falcon()` shell function. Add `--force` when replacing the earlier preview `.falconrc`. Setup writes a small `~/.falconrc`, creates a stable `~/.local/bin/falcon` launcher pinned to the Python environment where Falcon is installed, and installs a managed initialization block in the active shell's `~/.zshrc` or `~/.bashrc`. Falcon therefore remains available when another Conda environment—or `base`—is active. The block also enables dynamic completion. Use `--no-shell` to skip the rc-file change.

The image, pull secret, scheduler, kube-state-metrics endpoint, and container shell remain internal deployment defaults. Setup uses `LOGNAME` once to seed identity-based values in `.falconrc` and prompts for the namespace, comma-separated mount paths, optional comma-separated `KEY=VALUE` environment variables, and shared-memory percentage. Press Enter at any prompt to retain its displayed default:

- `LOGNAME=divyam.c` gives namespace `divyamc-dev`.
- User mount: `/media/beegfs/users/divyam.c/`.
- Team mount: `/media/beegfs/teams/`.

After setup, Falcon reads the namespace and all mount paths exclusively from `.falconrc`; changing `LOGNAME` does not change runtime behavior. Edit these persisted values directly when needed:

```yaml
cluster:
  namespace: divyamc-dev
runtime:
  volumes:
    - /media/beegfs/users/divyam.c/
    - /media/beegfs/teams/
  environment:
    WANDB_MODE: offline
    HF_HOME: /media/beegfs/users/divyam.c/.cache/huggingface
```

Environment variables in `.falconrc` are passed to every Falcon pod. They merge over Falcon's internal environment defaults, so an explicitly configured key wins.

The default config contains GPU eviction thresholds, dashboard display behavior, and `resources.shared_memory_percent: 15`. Dashboard sampling is fixed at a fast one-second target and is not a setup question. A preset can override the shared-memory percentage:

```yaml
presets:
  h100:
    gpu_type: h100
    minimum_utilization: 90
    shared_memory_percent: 20
```

## Dynamic GPU requests

There are no shell aliases and no fixed list of GPU counts. The CLI parses any positive `xN` suffix and validates it against the cluster:

```bash
falcon h100 -- python train.py
falcon h100x2 -- torchrun --nproc-per-node=2 train.py
falcon 2080tix3 -- python train.py
falcon a6000x2 -- python evaluate.py
```

If the largest matching node has four GPUs, `2080tix3` is valid and `2080tix5` is rejected immediately.

The preview wrapper's submission syntax is also accepted and translated into the same native planner:

```bash
falcon -j train-job -n 3 -g 2080ti -a -- python train.py
```

Here `-n` is the GPU count and `-g` accepts either a configured preset name or its GPU type. New scripts can prefer `falcon 2080tix3`, but both forms use identical planning and overrides.

Falcon reads allocatable resources, active pod requests, GPU products, and schedulability from the same internal metrics source used by `jet resources`. For `N` requested GPUs on a `T`-GPU node, CPU and RAM target `N/T` of the node. When the GPUs are currently free, each value is capped at the amount currently available. If the GPUs are busy, Falcon queues the proportional request instead of using an arbitrary small fallback.

Falcon uses the eligible GPU node with the most absolute free CPU and RAM to size the request, but does not add a hostname selector. Kubernetes remains free to place the pod on any node that satisfies its GPU type and resource requests. Use `--pin-node` only when explicit hostname placement is required.

## Shared memory and overrides

`/dev/shm` defaults to 15% of the final allocated RAM. For example, a `60Gi` RAM allocation receives `9Gi` shared memory.

Override it by percentage or exact size:

```bash
falcon 2080tix3 --shm-percent 25 -- python train.py
falcon h100 --shm-size 40Gi -- python train.py
```

CPU, RAM, job name, and raw Jet arguments remain overridable. Namespace has no CLI override; Falcon uses the value persisted in `.falconrc`:

```bash
falcon 2080tix3 -c 48:48 -m 50Gi:50Gi -j experiment -- python train.py
falcon h100 --jet-arg='--priority high-priority' -- python train.py
```

If an override is not currently schedulable, Falcon clearly warns that the job will remain pending. Use `--dry-run --explain` to inspect the generated YAML and Jet invocation without submitting.

Use `--max` when the job should eventually receive near-full proportional node resources rather than being reduced to what is free at submission time:

```bash
falcon 2080tix4 --max -- python train.py
```

Falcon requests 95% of the CPU and RAM share represented by the GPU count on the strongest schedulable matching node. On a four-GPU node with 96 CPUs, `2080tix4 --max` requests `91.2` CPUs even if only 60 are currently free. The job remains pending until a matching node can satisfy the request. Explicit `--cpu` or `--memory` values override `--max` for that resource.

Command jobs are followed and deleted after completion or interruption by default. Add `--async` to leave the submitted job running. Omitting the command launches a debug pod.

## Job controls

The native command includes the old wrapper's operational commands:

```bash
falcon logs [job]
falcon attach [job]
falcon top [job]
falcon delete [job ...]
falcon clean
```

When the job argument is omitted, Falcon uses the most recently launched or selected job. Tab completion lists live job names for `logs`, `attach`, `top`, and `delete`. Command, option, preset, and valid GPU-count completion are also dynamic.

Shell initialization embeds command, preset, and option candidates directly in Zsh or Bash, so ordinary Tab presses do not start Python or import the dashboard. GPU capacity suggestions are cached across shells for five minutes. Job-name completion queries Kubernetes only when needed and keeps a two-second in-shell cache.

Falcon passes the active `CONDA_PREFIX` or `VIRTUAL_ENV` to Jet as `--pyenv`. Jet mounts that environment and places its `bin` directory first on the pod's `PATH`; Falcon also sets Conda's standard `CONDA_AUTO_ACTIVATE_BASE=false` configuration variable so a freshly initialized interactive shell does not replace the selected environment with base. This does not suppress an explicit `conda activate base` command written by the user in shell startup files.

Completion can be inspected manually with `falcon completion zsh` or `falcon completion bash`.

## Dashboard

```bash
falcon dashboard
```

The dashboard follows nvitop's boxed-slot layout, with one responsive slot per Kubernetes Job. Jobs with multiple pods are aggregated before display. Every slot shows state, nodes, pod count, GPU type/count, live utilization, EMA, total VRAM, CPU, RAM, and age. GPU, VRAM, CPU, and RAM meters all use the same scale: green below 30%, yellow from 30–80%, and red above 80%. When GPU EMA falls below that GPU type's configured eviction floor, the entire job card becomes red.

CPU and RAM usage come from `kubectl top pods`; their percentages are calculated against container requests from running pods only. Failed retries and completed pods remain part of Job history but never inflate active CPU, RAM, or GPU totals. GPU utilization is weighted across sampled running GPUs and VRAM is summed across the active Job. Interactive dashboards and multi-sample agent snapshots keep one long-lived one-second `nvidia-smi` stream per running GPU pod instead of opening a Kubernetes exec session every frame. Pod inventory and CPU/RAM metrics are cached for five seconds because they change more slowly; `--job` scopes both Kubernetes queries to that job. GPU EMA starts with a five-sample arithmetic mean, then defaults to `ema_alpha: 0.1`, giving it roughly a 10-second smoothing time constant at the one-second sampling rate. Old generated `0.25`, `0.08`, and `0.02` values automatically migrate to the current default; other explicitly configured values remain overrides. The default eviction floors are 90% for H100 and 30% for A6000/2080Ti.

Cards and bars resize with the terminal. Wide screens show a two-column grid; compact screens keep the boxes and percentages but replace bars with numbers. Succeeded jobs are sorted to the bottom and collapse to compact one-line cards. When another row cannot fit completely, its leading edge remains visible and the header reports how many jobs are above or below. Only extremely small terminals below `42×14` show a resize message. Mouse-wheel events are captured inside the alternate-screen TUI, so they navigate jobs without exposing terminal scrollback. Use the wheel, `j`/`k`, or arrow keys to select; `Enter` opens nvitop only for non-succeeded jobs; `r` refreshes and `q` or `Ctrl+C` quits.

The dashboard is bounded for agents and scripts. When stdout is not a terminal, `falcon dashboard` automatically collects five one-second samples, prints one compact, ANSI-free line per job, and exits instead of emitting repeated TUI frames. In snapshot mode EMA is the mean across the complete requested sample window rather than the first frame. Use `falcon dashboard --once` to force that snapshot in a terminal, or `falcon dashboard --json` for structured output.

Agents can limit collection to one job and choose a longer observation window:

```bash
falcon dashboard --job <job-name> --json
falcon dashboard --job <job-name> --samples 15 --interval 1 --json
```

The default five samples balance stability and latency. Use 10–20 samples for bursty workloads, or `--samples 1` only when an instantaneous frame is explicitly required.
