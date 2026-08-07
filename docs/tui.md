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
