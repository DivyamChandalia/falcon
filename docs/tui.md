# TUI controls

Both TUIs require at least 80×22. Smaller terminals show a clean resize
message. Layouts adapt from 80×22 through wide monitoring displays. Dashboard
and Resources refresh on the same one-second cadence. Dashboard resource usage
keeps a readable fixed height; at the shortest supported heights Events is
temporarily hidden and returns automatically when space is available.
The Resources summary uses the same two-row header rhythm and keeps nodes,
running Jobs, CPU, memory, and GPU availability on one line.

## Jobs dashboard

- `Tab` / `Shift+Tab`: cycle panes
- `1` Jobs, `2` Resources, `3` Events, `4` Selected Job
- `↑` / `↓`, `j` / `k`: navigate the focused pane
- `PageUp` / `PageDown`, `Home` / `End`: page or jump
- `Enter` or `z`: expand; `Esc`: restore
- `/`: search; `f`: filters; `s`: sort
- `Space`: mark; `a`: mark all; `A`: clear marks
- `v`: choose visible panes
- `r`: refresh; `q`: quit

Events follow the newest entry until the user scrolls backward. New events do
not move a manually positioned viewport. `End` (or scrolling back to the last
page) resumes follow. Changing Jobs resets event position predictably.

## Resources

- `↑` / `↓`, `j` / `k`: select nodes
- `PageUp` / `PageDown`, `Home` / `End`: page or jump
- `Enter`: inspect the node and its consumers
- `Esc`: return to node list
- `←` / `→`: cycle Nodes, GPU Overview, and GPU Allocations (wraps)
- `v`: switch the namespace pie between requested GPU count and allocated VRAM
- Click a GPU Overview or GPU Allocations sub-pane to select it; `Enter`
  expands the selection and `Esc` restores it
- `r`: refresh; `q`: quit

The node list uses a single schedulability state (`Yes`, `Cordoned`,
`Not ready`, or `Unknown`). Infrastructure Pods are omitted from consumer
rows and Pod counts, while their requests remain included in headroom totals.
The header marks stale snapshots while retaining their last valid values.
GPU, CPU, memory, and Pod columns are compact and numeric values are
right-aligned. CPU headroom is shown consistently in decimal cores. The node
inventory keeps a fixed height so every row gets priority; the selected-node
panel expands or shrinks into the remaining space and is hidden only when it
reaches its minimum height. `Enter` still opens the full inspector. Hover the
selected-node panel to scroll its workload rows without moving to another
node. GPU VRAM is the memory of one device, not the sum across the node. Node
names use natural ordering, so `node10` follows `node9`.

GPU Allocations is scheduler-facing allocation accounting from the same local
resource snapshot as the other Resources views; it does not query or infer GPU
compute utilization. The history is collected in the background, is process
local, and is labeled `since launch` (up to 24 hours and 20,000 points). The
Dashboard and Resources screens share Falcon's true-colour semantic palette for
status, pressure, accents, and totals. Namespace slices use the dedicated
seven-colour colour-blind-friendly palette, with cyan reserved for accents and
totals.
Falcon pins the Rich console used by both apps to truecolour output. This is
intentional for tmux: a `screen-256color` `$TERM` must not downgrade these
explicit hex colours to xterm-256 or ANSI-16 values.
The filled namespace pie and Allocation History use one shared percentage
legend. The legend is a single vertical list in the left column beside
Allocation History; the pie is below them with GPU-requesting Pods to its right.
Names are not repeated inside or below either graph. The pie shows the six
largest visible namespaces, `Other`, and `System/hidden` when needed to
reconcile totals. Both graphs switch bases together: percentages are exactly
namespace requested GPU count divided by total requested GPU count in GPU mode,
or namespace allocated VRAM divided by total allocated VRAM in VRAM mode.
