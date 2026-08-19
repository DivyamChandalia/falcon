# TUI controls

Dashboard requires at least 80×22 and Resources requires at least 80×20.
Smaller terminals show a clean resize message. Layouts adapt from their minimum
sizes through wide monitoring displays. Dashboard and Resources refresh on the
same one-second cadence. Dashboard resource usage
keeps a readable fixed height; at the shortest supported heights Events is
temporarily hidden and returns automatically when space is available.
The Resources header keeps its view selector on the title row. The summary
keeps nodes, running Jobs, and free GPU/CPU/memory headroom on one line.

## Jobs dashboard

- `Tab` / `Shift+Tab`: cycle panes
- `1` Jobs, `2` Resources, `3` Events, `4` Selected Job
- `↑` / `↓`, `j` / `k`: navigate the focused pane
- `PageUp` / `PageDown`, `Home` / `End`: page or jump
- `Enter` or `z`: expand; `Esc`: restore
- `/`: search; `f`: filters; `s`: sort
- `Space`: mark; `a`: mark all; `A`: clear marks
- `k` / `F9`: kill the marked Jobs, or the selected Job when none are marked
- `c`: clean succeeded Jobs within the marked set; with no marks, clean all succeeded Jobs
- `v`: choose visible panes
- `r`: refresh; `q`: quit

Events follow the newest entry until the user scrolls backward. New events do
not move a manually positioned viewport. `End` (or scrolling back to the last
page) resumes follow. Changing Jobs resets event position predictably.

The expanded Selected Job inspector scrolls as one page, including Job details
and the full command. Use `↑`/`↓`, `PageUp`/`PageDown`, `Home`/`End`, or the
mouse wheel; resizing is not required to reach fields below the fold.

The expanded Resource Usage inspector also scrolls as one page. When the mouse
is over a GPU, VRAM, CPU, or RAM utilization card, the wheel moves through that
metric history instead. `←`/`→` always navigate history. The GPU Devices
section shows responsive per-device model, UUID, VRAM, utilization,
temperature, power, ECC, and driver columns. Active compute processes appear
indented directly below their GPU with PID, process name, GPU utilization, and
allocated VRAM in GiB.
Device metrics and per-process GPU utilization use persistent `nvidia-smi`
streams; process names and allocated VRAM are reconciled every five seconds.

## Resources

- `↑` / `↓`, `j` / `k`: select nodes
- `PageUp` / `PageDown`, `Home` / `End`: page or jump
- `Enter`: inspect the node and its consumers
- `s`: cycle selected-node workload sorting (Namespace, CPU, Memory, GPU)
- `Esc`: return to node list
- `←` / `→`: cycle Nodes and GPU Allocations (wraps)
- `v`: switch the namespace pie between requested GPU count and allocated VRAM
- Click a GPU Allocations sub-pane to select it; `Enter` expands the selection
  and `Esc` restores it
- `r`: refresh; `q`: quit

The node list uses a single schedulability state (`Yes`, `Cordoned`,
`Not ready`, or `Unknown`). Infrastructure Pods are omitted from consumer
rows and Pod counts, while their requests remain included in headroom totals.
The header marks stale snapshots while retaining their last valid values.
The selected node row is highlighted across its full width without replacing
the resource-pressure foreground colours. The inventory columns are Node,
CPUs, RAM (GB), GPUs, GPU Type, and Sched. CPU, RAM, and GPU free/allocatable
bars expand into all remaining width with uniform column padding and use the
Dashboard's shared green/yellow/red headroom thresholds. Per-model GPU
availability remains right-aligned in the same top summary position on both views.
Inspector resources remain ordered CPU, memory, then GPU. CPU
headroom is shown consistently in decimal cores. The node
inventory keeps a fixed height so every row gets priority; the selected-node
panel expands or shrinks into the remaining space and is hidden only when it
reaches its minimum height. `Enter` still opens the full inspector. Hover the
selected-node panel to scroll its workload rows without moving to another
node. GPU VRAM is the memory of one device, not the sum across the node. Node
names use natural ordering, so `node10` follows `node9`.

Selected-node workloads are shown as Namespace, Job, Status, CPU, RAM, and GPU.
The `s` sort preference is stored in `resources.consumer_sort` and is restored
on the next launch. CPU, memory, and GPU sorts show the largest requests first;
Namespace sorts naturally by namespace, then Job.

GPU Allocations is scheduler-facing allocation accounting from the same local
resource snapshot as the other Resources views; it does not query or infer GPU
compute utilization. Opening Resources starts one detached collector for the
configured resource endpoint. It continues after the TUI closes and stores up
to 24 hours and 20,000 snapshots in
`$XDG_STATE_HOME/falcon/resources-history-*.sqlite3` (or
`~/.local/state/falcon`). Later TUI launches load that window immediately. The
Dashboard and Resources screens share Falcon's true-colour semantic palette for
status, pressure, accents, and totals. Namespace slices use the dedicated
seven-colour colour-blind-friendly palette, with cyan reserved for accents and
totals.
Falcon pins the Rich console used by both apps to truecolour output. This is
intentional for tmux: a `screen-256color` `$TERM` must not downgrade these
explicit hex colours to xterm-256 or ANSI-16 values.
Use `--color=truecolor` (also the default) or `FALCON_COLOR=truecolor` to make
the choice explicit. `--color=256` and `--color=16` are opt-in fallbacks;
`--color=auto` still chooses truecolour for tmux-like terminals and only
falls back automatically for `TERM=dumb`. Set `FALCON_COLOR_DEBUG=1` to log
the selected mode, the framework-detected mode, and the exact RGB tuple.
The filled namespace pie and Allocation History use one shared percentage
legend. The legend is a single vertical list in the left column beside
Allocation History; the pie is below them with GPU-requesting Jobs to its right.
Names are not repeated inside or below either graph. The pie shows the six
largest visible namespaces, `Other`, and `System/hidden` when needed to
reconcile totals. Both graphs switch bases together: percentages are exactly
namespace requested GPU count divided by total requested GPU count in GPU mode,
or namespace allocated VRAM divided by total allocated VRAM in VRAM mode.

The current step-line chart reflects that scheduler allocations change in
discrete steps. Its rasterizer merges connectivity where series overlap, so
rises, falls, and intersections remain continuous box-drawing paths. Other
useful designs are a stacked step-area chart (best for
showing both total pressure and namespace share), time-bucketed stacked bars
(clearest for long windows), a namespace-by-time heatmap (best with many
namespaces), and small-multiple sparklines (best for comparing shapes without
stacking).
