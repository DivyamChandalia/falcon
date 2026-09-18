# TUI controls

Dashboard requires at least 80×22 and Resources requires at least 80×20.
Smaller terminals show a clean resize message. Layouts adapt from their minimum
sizes through wide monitoring displays. At `160×30` and larger, Dashboard uses
two equal-width columns: Jobs occupies the upper half of the left column,
Events occupies the remaining left-column space, Selected Job fills the right
column above Resources, and Resources remains six rows high at the bottom.
Below that breakpoint Dashboard keeps its stacked layout. Dashboard and
Resources refresh on the same one-second cadence. At the shortest supported
heights Events is temporarily hidden and returns automatically when space is
available.
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
- `c`: clean succeeded Jobs within the marked set; with no marks, clean all succeeded Jobs. It switches Logs between full height and a two-line view only when the nested Logs viewport has keyboard focus.
- `v`: choose visible panes
- `r`: refresh; `q`: quit

Events follow the newest entry until the user scrolls backward. New events do
not move a manually positioned viewport. `End` (or scrolling back to the last
page) resumes follow. Changing Jobs resets event position predictably.

The Jobs table keeps the GPUs column visible in the minimum supported pane and in
the half-width Dashboard layout. When space is tight, Active Pod and Age yield
before the GPU column; NODE and GPUs appear together once the pane is
wide enough for both.

The Selected Job inspector keeps compact two-column details above a
full-height Logs viewport. In the wide two-column layout it is shown
automatically; `Enter` still makes it the exclusive full-screen pane and `Esc`
returns to the responsive layout. A small, left-aligned Command row sits below
RAM request, aligned with the metadata values; its copy icon is in the same
value column as entries such as RAM and age. The label is informational and
does not select a second pane. Logs are full-height by default and `c` switches
them to a two-line viewport. `Ctrl/Cmd+C` copies the selected Logs viewport.
The log viewport opens at the newest line and follows new output; `Home` or
scrolling upward pauses follow until `End` resumes it. Click the Logs viewport
to select it; it owns the keyboard and mouse-wheel scrolling while the outer
Selected Job pane continues to select Jobs with `↑`/`↓`.
Running Pods stream non-interactive `attach` output; for
succeeded or failed attempts Falcon loads the equivalent of
`falcon logs --no-follow --tail 200`. `←`/`→` switches between Pod attempts,
with the newest active Pod selected initially. Captured output is bounded to
200 lines per Pod and expires from memory after 24 hours.

The expanded Resource Usage inspector also scrolls as one page. When the mouse
is over a GPU, VRAM, CPU, or RAM utilization card, the wheel moves through that
metric history instead. Its four metric cards always remain in a two-by-two
quadrant grid in the expanded view; larger terminals give each quadrant more
room rather than changing to a vertical stack. `←`/`→` always navigate history.
The GPU Devices
section shows responsive per-device model, UUID, VRAM, utilization,
temperature, power, ECC, and driver columns. Active compute processes appear
indented directly below their GPU with PID, process name, GPU utilization, and
allocated VRAM in GiB.
Device metrics and per-process GPU utilization use persistent `nvidia-smi`
streams; process names and allocated VRAM are reconciled every five seconds.

## Resources

- `↑` / `↓`, `j` / `k`: navigate the active side
- `PageUp` / `PageDown`, `Home` / `End`: page or jump
- `Enter`: inspect the node and its consumers
- `s`: cycle the shared workload sorting (Namespace, CPU, Memory, GPU) for
  Selected Node and GPU-requesting Jobs
- `Esc`: return to node list
- Below `160×30`, `←` / `→` cycle Nodes and GPU Allocations (wrapping). At
  `160×30` and above, both sides remain visible and `←` / `→` do nothing.
- In the wide layout, `Tab` follows Allocation History, Namespace Pie,
  GPU-requesting Jobs, Nodes, and Selected Node; `Shift+Tab` reverses that
  order. The active side is retained across resizes.
- `m`: cycle the allocation charts through GPU, scheduler-requested memory, and
  scheduler-requested CPU cores
- `v`: while the GPU mode is active, switch between requested GPU count and
  allocated VRAM. It does not leave or enter GPU mode.
- `l`: toggle Allocation History between linear and logarithmic y-axis scale
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
on the next launch. It is shared by Selected Node workloads and GPU-requesting
Jobs, including their expanded views, so changing it in either pane updates
the other. CPU, memory, and GPU sorts show the largest requests first;
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
At `160×30` and above, Resources keeps its header, summary, controls, and
footer full width. The node inventory and Selected Node details share the left
half; GPU Allocations occupies the right half. The node details hide when the
inventory needs the available height. `Enter` expands the active allocation
panel or Selected Node to the full Resources body, and `Esc` restores the
combined layout.

The filled namespace pie and Allocation History use one shared percentage
legend. In GPU Allocations, the left stack is a fixed 24-column Namespace
Legend followed by the pie; the legend's outer panel matches the taller
Allocation History panel on the right, while its rows remain content-driven.
Allocation History independently spans from the legend's right edge to the
full pane width. GPU-requesting Jobs starts after the pie's aspect-correct
footprint, so its narrower column never constrains the history chart. Long
namespace names are ellipsized. The pie uses the terminal's approximately
2:1 cell aspect ratio and gives any excess width to the history/jobs stack.
Even the compact minimum Resources pane retains a drawable pie footprint. The
legend uses every available row for namespaces and adds `Other` only when
there are more categories than the current pane can display. The same aligned
legend is retained when History or Pie is expanded. The view selector is
hidden while a panel is expanded; `Esc` restores it with the combined/page
layout. Names are not repeated inside or below either graph. `System/hidden`
is retained when needed to reconcile totals. Both graphs switch bases together
within the active pair. GPU mode is exactly namespace requested GPU count
divided by total requested GPU count, VRAM mode is namespace allocated VRAM
divided by total allocated VRAM, and the `m` pair uses scheduler-requested
memory (GiB) or CPU cores for every active workload. The GPU and CPU/memory
history series are additive to the existing service history format, so an
older service remains readable; its historical CPU and memory columns begin
collecting after the service is relaunched.

The current step-line chart reflects that scheduler allocations change in
discrete steps. Press `l` to use a `log1p` y-axis when large allocations would
flatten smaller changes; zero remains at the baseline. Its rasterizer merges
connectivity where series overlap, so
rises, falls, and intersections remain continuous box-drawing paths. Other
useful designs are a stacked step-area chart (best for
showing both total pressure and namespace share), time-bucketed stacked bars
(clearest for long windows), a namespace-by-time heatmap (best with many
namespaces), and small-multiple sparklines (best for comparing shapes without
stacking).
