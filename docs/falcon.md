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

`falcon clean` deletes succeeded Jobs only; failed and running Jobs are retained for inspection. The dashboard exposes the same operation with `c`: it lists all succeeded Jobs in a dedicated confirmation dialog, then cleans them with `Enter` or `y`.

When the job argument is omitted, Falcon uses the most recently launched or selected job. Tab completion lists live job names for `logs`, `attach`, `top`, and `delete`. Command, option, preset, and valid GPU-count completion are also dynamic.

Shell initialization embeds command, preset, and option candidates directly in Zsh or Bash, so ordinary Tab presses do not start Python or import the dashboard. GPU capacity suggestions are cached across shells for five minutes. Job-name completion queries Kubernetes only when needed and keeps a two-second in-shell cache.

Falcon passes the active `CONDA_PREFIX` or `VIRTUAL_ENV` to Jet as `--pyenv`. Jet mounts that environment and places its `bin` directory first on the pod's `PATH`; Falcon also sets Conda's standard `CONDA_AUTO_ACTIVATE_BASE=false` configuration variable so a freshly initialized interactive shell does not replace the selected environment with base. This does not suppress an explicit `conda activate base` command written by the user in shell startup files.

Completion can be inspected manually with `falcon completion zsh` or `falcon completion bash`.

## Dashboard

```bash
falcon dashboard
```

The full-screen dashboard uses a pure-black nvitop-inspired layout: header and clock, cluster-state summary, search/filter/sort controls, a responsive Jobs table, selected-Job pane, resource history, events, and a context-sensitive footer. Namespace and cluster configuration are never rendered. Wide terminals show `MARK | NAME | STATUS | ACTIVE POD | NODE | GPU TYPE | AGE`; GPU type and then node disappear as width decreases, while identity, status, active-pod state, and age remain. GPU allocations use compact names such as `2080tix1`, `a6000x2`, and `h100x1`.

The supported minimum terminal is `80×30`. Below that size Falcon shows an explicit resize message instead of clipping Jobs or making Events unreachable. At or above the minimum, at least four Job rows remain usable. The summary is a single row with a fixed lower border: Job state is on the left, followed by one `RESOURCES AVAILABLE` label and cluster-wide `free/total` counts for `2080Ti`, `A6000`, and `H100`. Falcon obtains these counts from the same metrics endpoint and parser as `jet r`, refreshing every 15 seconds. GPU counts are green below 30% used, yellow from 30–80%, and red above 80%. Events has a stable five-row viewport, so selecting Jobs never randomly resizes the Jobs pane.

The cursor and htop-style marks are independent and stored by Kubernetes Job UID. `Space` marks, `Shift+Space` marks and advances, `a` marks visible Jobs, `A` clears marks, and `m` toggles marked-only filtering. `/` searches Job identity/state/node/GPU fields, `f` opens filters for status, active pod, node, GPU, and marks, and `s` cycles sort fields. `k` or `F9` opens a guarded Job/pod deletion dialog; batch and large-batch deletion require additional confirmation, and rows remain until Kubernetes confirms deletion.

`Tab` and `Shift+Tab` cycle Jobs, Selected Job, Resource Usage, and Events; arrow keys always navigate the focused pane’s contents. When a pane is expanded, those keys switch the full-screen expanded page rather than focusing a hidden pane. `1`/`2`/`3` jump directly to Jobs/Resources/Events, while `4` focuses Selected Job. `Enter` or `z` expands the focused pane and `Esc` restores the layout. Expanded Selected Job shows lifecycle timestamps, active pod and node, GPU/CPU/RAM allocation, restarts, completions, EMA/risk state, and a scrollable command viewport. Use the mouse wheel, `↑`/`↓`, `PgUp`/`PgDn`, and `Home`/`End` to move through a long command. Mouse scrolling in expanded Jobs moves only the table viewport and never changes the selected Job; expanded Resources and Events scroll their own history and event positions. Each pane retains its cursor or history offset. Events consolidate repeated counts, follow new entries until manually scrolled, and support pane-local search.

Expanded Resource Usage automatically selects a wide diagnostic layout, a narrower compact layout, or a vertically collapsed summary according to the current terminal dimensions. Wide mode shows four full-width stacked GPU, VRAM, CPU, and RAM sections. Each section keeps left-aligned statistics in a narrow column and gives the remaining width to a right-aligned, bottom-aligned history graph; compact modes use the same alignment in a 2×2 card grid. On terminals that confirm synchronized-output support, graphs scale vertically to the statistics area. Falcon automatically falls back to a single bottom-aligned magnitude row when synchronized output is unavailable, preventing multi-row repaint tearing over SSH, tmux, or unsupported terminal emulators. Graphs contain no sample-count or collection labels. GPU and VRAM are sampled every second: at `100%`, one bar is one metric sample; at `50%`, each bar averages two samples; at `25%`, each bar averages four. CPU and RAM use the five-second Kubernetes metrics cadence, so one native sample is a five-column plateau at `100%` and is averaged into the corresponding time buckets when zoomed out. Terminal width determines how many bars and therefore how much elapsed time fits. Percentages, history blocks, and device utilization all use the same bands: below 30% green, 30–79% yellow, and 80% or above red. High utilization is color only, never a warning. The red `! EVICTION RISK` warning appears only after the complete 60-sample GPU average is below that GPU type’s configured threshold. Missing GPU telemetry is labelled unavailable rather than rendered as zero. Use arrows or `h`/`l` to move through history, `R` to cycle the history range, `+` to zoom in, `-` to zoom out, `Z` to cycle zoom levels, and `Esc` to restore the normal dashboard. The pinned footer displays the current zoom percentage; expanded-pane shortcuts are never duplicated in pane subtitles. Completed Jobs retain the last valid sample. API failures retain prior values and mark them stale instead of clearing the screen.

Press `f` while Jobs is focused to open filters. Use `↑`/`↓` to select Status, Pod, Node, GPU, or Marked; use `←`/`→` or `Space` to change the bracketed value; then press `Enter` to apply or `Esc` to cancel. The Jobs footer always exposes the `f Filters` shortcut.

CPU and RAM usage come from `kubectl top pods`; their percentages use requests from running pods only. Interactive dashboards and multi-sample agent snapshots keep one long-lived one-second `nvidia-smi` stream per running GPU pod instead of opening a Kubernetes exec session every frame. Pod inventory, CPU/RAM metrics, and selected-Job events are cached for five seconds; `--job` scopes pod and CPU/RAM queries. GPU EMA remains a visible trend metric: it starts with a five-sample arithmetic mean and then uses `ema_alpha: 0.1`. Eviction risk no longer uses that EMA. It requires a complete 60-sample rolling arithmetic average and compares only that average with the 90% H100 or 30% A6000/2080Ti floor, so a one-second utilization drop cannot trigger a warning. Agent snapshots use their complete requested sample window as the risk-average window.

The dashboard is bounded for agents and scripts. When stdout is not a terminal, `falcon dashboard` automatically collects five one-second samples, prints one compact, ANSI-free line per job, and exits instead of emitting repeated TUI frames. In snapshot mode EMA is the mean across the complete requested sample window rather than the first frame. Use `falcon dashboard --once` to force that snapshot in a terminal, or `falcon dashboard --json` for structured output.

Agents can limit collection to one job and choose a longer observation window:

```bash
falcon dashboard --job <job-name> --json
falcon dashboard --job <job-name> --samples 15 --interval 1 --json
```

The default five samples balance stability and latency. Use 10–20 samples for bursty workloads, or `--samples 1` only when an instantaneous frame is explicitly required.
