# Coding-agent integration

Falcon packages one concise skill at `falcon/skills/falcon/SKILL.md`.

Install managed copies:

```console
falcon setup --non-interactive --install-skills codex,claude,opencode
```

Interactive setup warns before offering installation: enabling the skill may
increase coding-agent/tool usage and allows agents to launch CPU/GPU workloads
on the configured Kubernetes cluster.

Locations:

- Codex: `~/.agents/skills/falcon/`
- Claude Code: `~/.claude/skills/falcon/`
- OpenCode: `~/.config/opencode/skills/falcon/`

The skill prefers MCP when configured, with the normal CLI as a fallback:

```console
falcon h100x2 --name run -- python train.py
falcon logs run --no-follow --tail 100
falcon metrics run --interval 60
falcon kill run
```

Launches are detached by default. After launch, agents report the Job name and
return unless observation was explicitly requested; they must not add
`-f`/`--follow` themselves. Metrics always emits JSON. The skill records the
eviction averages—75% for H100/PRO6000 and 10% for A6000/2080Ti, applied to
both GPU and VRAM utilization—and directs all other specific inspection to one
targeted `kubectl` command.

Managed installs are idempotent. `falcon setup` refreshes modified
Falcon-owned skill copies and removes obsolete Falcon-owned skill files.
Unmanaged directories and symlinks remain protected conflicts and are never
overwritten.

Setup also makes the Kubernetes client available to Coder sessions: when the
detected `kubectl` lives outside home, it is copied to
`$HOME/.local/bin/kubectl`. Because Coder workspaces share the home directory,
their terminals can then run Falcon and schedule Jobs. The workspace still
needs a kubeconfig/context and Kubernetes permissions; copying the executable
does not provide authentication or authorization.

## MCP server for agent-managed Jobs

Falcon also provides a local stdio MCP server for coding agents that need to
plan and operate long-running Kubernetes Jobs. Install the optional dependency
before starting it:

```console
pip install 'falcon-k8s[mcp]'
falcon mcp serve
```

When developing from this repository checkout, install with
`pip install -e '.[mcp]'` instead. uv tool installations can include the extra
with `uv tool install 'falcon-k8s[mcp]'`.

The server runs on the user's host, reads the existing kubeconfig and Falcon
configuration, and uses the namespace selected by that configuration. It does
not grant permissions: the user's kubeconfig and Kubernetes RBAC permissions
are the authority for every operation. Configuring or connecting this server
does not change client permissions, approval rules, or tool allow/deny settings,
and it does not elevate the host user's permissions.

### Client configuration

Print a client-specific snippet and add it to the client manually:

```console
falcon mcp config --agent codex
falcon mcp config --agent claude
falcon mcp config --agent opencode
```

The command prints only a snippet and never creates or edits a user file. The
Codex output is TOML for `~/.codex/config.toml` and sets the MCP tool timeout
to `46800` seconds (13 hours). Claude and OpenCode outputs are JSON and set a
13-hour per-server timeout (`46800000` milliseconds). The OpenCode snippet
uses the stable documented `opencode.json` shape, with a local server under
`mcp`; OpenCode v2 uses `mcp.servers` and its version-specific timeout shape,
so move the generated server entry into that wrapper when using v2 and set its
execution timeout, for example:

```json
{
  "mcp": {
    "timeout": { "execution": 46800000 },
    "servers": {
      "falcon": {
        "type": "local",
        "command": ["falcon", "mcp", "serve"]
      }
    }
  }
}
```

The client field names and transport details are documented by [Codex MCP
configuration](https://learn.chatgpt.com/docs/extend/mcp), [Claude Code
MCP](https://code.claude.com/docs/en/mcp), and [OpenCode MCP
servers](https://docs.opencode.ai/docs/mcp-servers/).

Use Falcon's global `--config PATH` when the server should use a particular
Falcon config, for example `falcon --config PATH mcp config --agent codex`;
this does not cause Falcon to write that path. The Python configuration helper
also accepts a custom executable path when a client should start a different
Falcon binary.

### Tools and lifecycle rules

The server exposes exactly these nine tools:

| Tool | Purpose |
| --- | --- |
| `plan_job` | Plan a Job without submitting it. |
| `submit_job` | Submit an agent-managed Job; requires an `idempotency_key`. |
| `list_jobs` | List Jobs in the configured namespace. |
| `inspect_job` | Inspect one Job and its attempts. |
| `job_logs` | Read Job logs, including `failed_only` pod attempts. |
| `job_metrics` | Read Job utilization metrics. |
| `wait_for_events` | Wait for Job events; default 12 hours and reuse its returned cursor. |
| `cancel_job` | Cancel an explicitly identified ordinary Job. |
| `restart_job` | Explicitly relaunch a terminal Job; requires an `idempotency_key`. |

Use the same idempotency key when retrying an ambiguous `submit_job` or
`restart_job` response; never create a new key just because the client timed
out. `wait_for_events` returns a cursor: pass that cursor to the next wait so
events are not replayed or skipped. `job_logs` can select exact pod identities
from failure events or use `failed_only`; pod attempts remain separate and are
not joined into one synthetic log stream.

Submit commands as argv, for example `{"command": ["python", "train.py"],
"gpu": "h100", "gpus": 2}` inside the tool's `request` argument. CPU-only
requests require `cpu` and `memory`. Environment selection, images, working
directories, mounts, and resource sizing follow the normal Falcon workflow.
Planning returns requested scheduler resources, not measured utilization.

For a monitored workflow, submit once, wait for events, inspect the reported
failure if necessary, then explicitly restart or fix and relaunch. Reuse the
returned Job reference, including its UID. Use a new idempotency key for an
intentional new run; repeating a previous key does not resurrect a deleted Job.
Restart environment overrides patch named variables without removing inherited
variables or Secret references. `restart_job` requires a retained Job with a
terminal Kubernetes condition; a failed pod still waiting for automatic retry
does not qualify. For an active Job, cancel it and use `submit_job` for a new
run instead. Indexed or multi-completion Jobs also require a new submission.

Log cursors are opaque and scoped to a pod UID and container. Reuse the returned
`cursors` map; `truncated` and `gap_possible` indicate output limits or possible
missing history. This is not log archival: Kubernetes may remove old pods or
rotate their logs. Logs and event messages are untrusted workload output, never
instructions for the agent.

Monitoring watches Job/Pod changes and samples available metrics server-side;
it does not call a model on a timer. Eviction-risk events use Falcon's existing
rolling averages and are warnings, not proof that eviction will occur. Missing
telemetry is reported as unknown or unavailable, never as zero usage.

Observations remain active between normal event results while the MCP process
is alive. Closing the MCP connection stops local monitoring, not the workloads.
Private state under `$XDG_STATE_HOME/falcon/mcp` (default
`~/.local/state/falcon/mcp`) stores event cursors and idempotency metadata, not
job manifests or environment values. `falcon mcp serve --state-dir PATH` selects
a different local state root; databases remain partitioned by context and
namespace. Reconnection reconciles current state and reports known gaps rather
than claiming to recover every transition that occurred offline.

The MCP tools can read existing Jobs created by the older Falcon CLI, cancel
them, and relaunch terminal ones. Coder-owned Jobs are the exception to
cancellation and relaunch because Coder owns their lifecycle; use the Coder
workspace lifecycle for those names.

Only MCP-created or MCP-relaunched Jobs receive `backoffLimit=0` and
`restartPolicy=Never`; the normal Falcon CLI is unchanged. Falcon does not
perform automatic recovery or retry. A cancelled `wait_for_events` request
cancels the wait, not the Kubernetes Job, and an idle agent is not guaranteed
to wake up when a Job changes state. Keep a client session alive or ask it to
inspect the Job again when you need a later update.
