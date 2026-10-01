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
- `↑` / `↓`, `j`: navigate the focused pane
- `PageUp` / `PageDown`, `Home` / `End`: page or jump
- `Enter` or `z`: expand; `Esc`: restore
- `/`: search; `f`: filters; `s`: sort
- `Space`: mark; `a`: mark all; `A`: clear marks
- `k` / `F9`: open kill/restart actions for marked Jobs, or the selected Job when none are marked; `k` works from any Dashboard pane, including Logs
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
them to a two-line viewport. Drag across log text with the mouse, then press
`Ctrl/Cmd+C` to copy only the selected text; with no text selected, the shortcut
keeps its existing whole-log copy behavior.
On macOS Terminal, use `Ctrl+C` or a terminal with OSC52 clipboard support;
macOS Terminal reserves `Cmd+C` and does not accept OSC52. Falcon also uses
`pbcopy`, `wl-copy`, `xclip`, or `xsel` automatically when one is available on
the host running the Dashboard.
The log viewport opens at the newest line and follows new output; `Home` or
scrolling upward pauses follow until `End` resumes it. Click the Logs viewport
to select it; it owns the keyboard and mouse-wheel scrolling while the outer
Selected Job pane continues to select Jobs with `↑`/`↓`.
Running Pods stream non-interactive `attach` output; for
succeeded or failed attempts Falcon loads the equivalent of
`falcon logs --no-follow --tail 200`. `←`/`→` switches between Pod attempts,
with the newest active Pod selected initially. Captured output is bounded to
200 lines per Pod and expires from memory after 24 hours.

With Selected Job focused, `Ctrl+T` opens a separate interactive `kubectl exec`
session in the selected running Pod/container, replacing the log viewport with
a terminal. If the selected attempt has finished, Falcon uses its newest running
attempt. The shell prefers your locally configured zsh or bash, falling back to
bash, zsh, then sh if unavailable in the container. It uses the same
startup wrapper as debug Jobs: it sources your detected rc file if accessible
inside the Pod, otherwise the container's home rc file. It preserves the Pod's
active Conda/virtual environment. The compact prompt shows only the current
directory's basename, for example `sigliprfdetr%`. Falcon does
not copy host rc files into the Pod. A fallback message identifies the shell
used when your preferred shell is unavailable. Kubernetes exec permissions still
apply.

Press `Ctrl+T` again for another window. `←`/`→` navigates Pod logs followed by
terminal windows; `Alt+←`/`Alt+→` sends cursor movement to the shell. Other typing,
paste, Tab, and `Ctrl+C` go to the shell, so Dashboard shortcuts do not interfere.
`Ctrl+D` (or `Ctrl+W`) closes
the active terminal and returns to logs. Closing a terminal or the Dashboard
ends its exec connection, not the Kubernetes Job. Sessions remain connected when
you switch windows or Jobs, but are not persisted after Dashboard exit. At most
32 terminal windows may be open. Embedded terminals require a POSIX host and
an available shell in the container. A visible block cursor marks the input
position. Mouse-wheel scrolling retains up to 200 lines of terminal output;
typing returns to the live cursor. Scrolling upward pauses follow until you
scroll back down or type. This terminal scrollback is separate from Pod logs.
The copy button copies the active terminal's retained output as plain text;
when viewing Pod logs, it copies those logs instead.
Select terminal text with the mouse and press `Ctrl/Cmd+C` to copy that selection
without interrupting the shell. With no selection, `Ctrl+C` interrupts the
running command as usual.
Click Selected Job's metadata or outer frame to return to Dashboard shortcuts;
the footer switches to Dashboard commands while the terminal stays connected.
Click inside the terminal again to resume shell input and terminal controls.

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
- `s`: cycle the shared workload sorting (GPU, Memory, CPU, Namespace) for
  Selected Node and Namespace/Workload Allocation
- `Esc`: return to node list
- Below `160×30`, `←` / `→` cycle Nodes and Allocations (wrapping). At
  `160×30` and above, both sides remain visible and `←` / `→` do nothing.
- In the wide layout, `Tab` follows Allocation History, Namespace Pie,
  Namespace/Workload Allocation, Nodes, and Selected Node; `Shift+Tab`
  reverses that order. The active side is retained across resizes.
- `m`: cycle the allocation charts through GPU, scheduler-requested memory, and
  scheduler-requested CPU cores
- `v`: while the GPU mode is active, switch between requested GPU count and
  allocated VRAM. It does not leave or enter GPU mode.
- `l`: toggle Allocation History between linear and logarithmic y-axis scale
- Click an Allocations sub-pane to select it; `Enter` expands the selection
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
on the next launch. It defaults to GPU and is shared by Selected Node workloads
and Namespace/Workload Allocation, including their expanded views, so changing
it in either pane updates the other. The selected resource is primary; ties are
resolved by GPU, memory, CPU, then natural namespace/workload name priority.

Allocations is scheduler-facing allocation accounting from the same local
resource snapshot as the other Resources views; it does not query or infer GPU
compute utilization. Opening Resources starts one detached collector for the
configured resource endpoint. It continues after the TUI closes and stores up
to 24 hours and 20,000 snapshots in
`$XDG_STATE_HOME/falcon/resources-history-*.sqlite3` (or
`~/.local/state/falcon`). Later TUI launches load that window immediately. The
Dashboard and Resources screens share Falcon's true-colour semantic palette for
status, pressure, accents, and totals. Namespace slices use a dedicated
15-colour colour-blind-friendly palette before repeating, with cyan reserved
for accents and totals.
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
half; Allocations occupies the right half. The node details hide when the
inventory needs the available height. `Enter` expands the active allocation
panel or Selected Node to the full Resources body, and `Esc` restores the
combined layout.

When the combined layout is tall enough, Selected Node is capped at ten rows
and a separate lower-left row shows three cluster-wide namespace-share pies
for CPU, memory, and GPU. Allocation History and Namespace/Workload Allocation
then stack across the full right half; the ordinary single namespace pie is
used instead at smaller sizes. The three cluster pies are selectable by mouse
or `Tab`: selection switches the history metric and sorts both namespace
parents and workloads by that resource. `Enter` expands the selected pie with
the matching Namespace/Workload Allocation hierarchy on its right.

Press `f` to filter the Namespace/Workload Allocation tree by
minimum requested GPUs, CPU cores, memory, and a case-insensitive namespace
substring. Filters combine with AND and apply to individual workloads before
namespace totals are calculated. Blank fields impose no restriction; bare
memory numbers mean GiB (`8` equals `8Gi`). Enter applies, Esc cancels, and
Clear (or `Ctrl+R` in the dialog) removes all filters. Filters survive resizing/expansion for the current
Resources session; history, pies, and node inventory remain cluster-wide.

The large combined Resources layout uses one global focus across Nodes,
Selected Node, History, Allocation, and each namespace pie. Tab/Shift+Tab or
clicks move that focus; remembered chart selections are not highlighted when
another pane is active. Nodes/Allocations header labels are hidden in combined
and individually expanded layouts; compact pages retain the selector.

Allocations has three independent bordered surfaces without a redundant
outer frame: Allocation History above the aspect-correct namespace pie and
Namespace/Workload Allocation tree. The normal stack uses a balanced vertical
split between History and the lower pie/allocation row (preserving the pie's
minimum height in very short terminals). In the large combined layout, History
and Namespace/Workload Allocation split the right side vertically 50/50. The tree replaces the standalone
legend and flat GPU Jobs list. Each namespace parent uses the chart's stable
colour and shows the sum of requested GPU count, memory, and CPU for its visible
children; child rows show workload and node. It includes CPU-only and
memory-only workloads, and sorting applies consistently to parents and children
without combining unlike units. The view selector is hidden while one surface
is expanded; `Esc` restores it with the combined/page layout. `System/hidden`
is retained in charts when needed to reconcile totals. Both graphs switch
bases together within the active pair. GPU mode is exactly namespace requested GPU count
divided by total requested GPU count, VRAM mode is namespace allocated VRAM
divided by total allocated VRAM, and the `m` pair uses scheduler-requested
memory (GiB) or CPU cores for every active workload. The GPU and CPU/memory
history series are additive to the existing service history format, so an
older service remains readable; its historical CPU and memory columns begin
collecting after the service is relaunched.

Enter-expanded Selected Node retains the ordinary node facts and Resource
Consumers inspector. Allocation pies remain cluster-wide and live in the
combined Resources layout or their own expanded allocation view. The `m`, `v`,
and `l` allocation controls remain active while the Selected Node inspector is
expanded; they update the allocation view that appears when it is restored.

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
