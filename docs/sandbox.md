# Coding-agent sandboxes

`falcon sandbox` runs a live terminal command in an asynchronous Kubernetes
Job. It uses the configured runtime image, including Falcon's normal CUDA 13
image, rather than downloading agent tools or building a separate image.
Run the launcher/broker on Linux with access to the cluster's shared paths.

```console
falcon sandbox --name agent-loop -c 4 -m 8Gi \
  --read-only /path/to/datasets --read-only /path/to/reference -- codex
falcon sandbox --name claude-loop --agent-bin /path/to/claude -- claude
falcon sandbox --name opencode-loop -- opencode --model provider/model
```

Everything after `--` belongs to the agent.

Tab completion supports sandbox options, `status`/`resume`/`stop`, and locally
registered session names whose Jobs still exist, excluding stopped sessions.
Completion uses a bounded cluster lookup rather than listing old local runs.
Directory completion after `--read-only` supports
absolute paths, `~/...`, and relative paths such as `../datasets`. Relative
mount paths resolve from the directory where you launch Falcon. Refresh your
current shell's completion after upgrading:

```bash
eval "$(falcon completion bash)"  # Bash; use zsh instead for Zsh
```

The default Job/session name is `falcon-{harness}-{working-directory-name}`,
for example `falcon-codex-jet-k8s`. Names are lowercased and punctuation is
normalized to hyphens; names longer than 63 characters receive a short hash
suffix. `--name` overrides this default. An existing session is not overwritten;
attach/resume it, stop it, or use a different explicit name. When its Job is
missing, failed/stopped local records are archived on the next launch so the
name can be reused without deleting history. Sessions with child Job records
must be stopped first. Authentication failures/timeouts never count as a
missing Job. `status` reports missing Jobs rather than failing with NotFound.

CPU-only sessions and child Jobs default to 1 CPU and 1 GiB; GPU options use
Falcon's normal resource planner. The active Conda/virtualenv is mounted read-write by
default. Use `--environment /path/to/env` to select an environment explicitly, or `--environment none`
to use image tools only.

Parent and child Jobs use that same environment for `python`, `python3`, and
`python -m pip`. Installs, upgrades, and removals modify the host environment
directly and remain after the Job stops. No per-sandbox environment is created.
Any separate virtualenv base interpreter stays read-only. Concurrent package
changes affect host processes and every child using the environment, so install
dependencies before starting concurrent workloads. With no active environment,
image tools are used.

Falcon's credential-bearing host broker must run from a separate Python
installation, outside the writable environment. If the active environment also
runs Falcon, launch Falcon from a dedicated installation or choose a different
environment with `--environment`. This prevents the agent from replacing the
broker's interpreter or imported packages.

No broad configured home/team mount is inherited automatically. Installations and paths must be
available at the same locations on eligible cluster nodes; laptop-only paths
are not uploaded or synchronized.

## Terminal and lifecycle

The default opens tmux in the current terminal. **Ctrl+B, D** detaches without
stopping the Job; closing the terminal also leaves it running. `--detach`
launches without attaching. Use `--detach --output json` for scripts.

**Ctrl+C stops the attached agent session**, returns you to your local shell,
and the launch-machine broker deletes the sandbox Job. Exiting the agent
normally also closes the terminal session and triggers that cleanup. Ctrl+C
is reserved for stopping the session, not cancelling an individual agent turn.
Working files and history are preserved; child Jobs are not automatically
deleted. Use `falcon sandbox stop NAME` to clean up those children too.
Automatic deletion requires the launch-machine broker to be connected.

```console
falcon attach agent-loop
falcon sandbox status agent-loop
falcon sandbox --output json status agent-loop
falcon sandbox resume agent-loop
falcon logs agent-loop --no-follow --tail 100
falcon sandbox stop agent-loop
```

`status` shows a readable summary by default: workspace, Job/agent/broker
state, and child Jobs. Put `--output json` before `status` for machine-readable
output. Historical sessions can still be inspected by typing their name.

Logs contain startup diagnostics; interact with the agent and its scrollback
through tmux. `stop` removes the session and its owned child Jobs, preserving
working files and shared agent history. Ordinary `falcon kill` remains a
job-only operation. Deleting the parent does not resurrect it or delete its
children automatically.

Sessions remain preemptable. A replacement Pod recreates tmux and resumes
the exact recorded agent conversation, with up to three Job retries. Codex
conversation IDs are attributed to the agent process's open rollout file;
Claude sessions receive an explicit ID; OpenCode uses a local session server
to create a known ID before opening the TUI. Missing/ambiguous IDs require
manual recovery: Falcon never resumes an unrelated shared `--last` session.
Ordinary agent exits stop the parent Job and are not retried. Arbitrary loop scripts can run but are
not automatically replayed after Pod loss. An exhausted Job must be stopped
and relaunched with the agent's explicit resume command.

### Continue after Codex usage limits reset

Opt in when launching a Codex sandbox:

```console
falcon sandbox --auto-continue -- codex
# Start your task, then detach with Ctrl+B, D when appropriate.
falcon sandbox status falcon-codex-WORKING-DIRECTORY
```

The in-Pod supervisor recognizes Codex's usage-limit notice and reads structured
quota windows through `codex app-server`. It waits until all exhausted windows
reset, adds a ten-second safety margin, and confirms quota is available before
typing `continue` into the original agent pane. Status shows the reset time and
whether automatic continuation is waiting or paused. This works without an
attached terminal; the launch-machine broker must still run for restricted child
Jobs and normal cleanup.

Automatic input is sent **only while detached**, outside tmux copy mode, with an
empty Codex input field (a dim/gray placeholder is allowed). An unfinished draft,
unknown composer styling, changed pane, unavailable quota information, billing
limit, or unsupported app-server leaves the session untouched. Checks are bounded
and quota queries are at most once per minute. There is at most one automatic
submission per reset window; state is retained across Pod replacement. Manual
progress that clears the limit notice cancels a pending continuation.

This option currently supports **Codex only** and applies to newly launched
sessions, not already-running Jobs. It never buys credits, redeems reset credits,
changes models, or modifies provider limits. Stopping or exiting the sandbox stops
the watcher. The detection uses Codex's visible limit notice, so future terminal
UI changes may require a Falcon update; quota reset times are not guessed from
human-readable dates. See the official [Codex app-server quota interface](https://learn.chatgpt.com/docs/app-server).

## Tools, permissions, and persistent settings

Falcon resolves the local executable and mounts its installation read-only.
Native release directories and npm packages with their Node runtime are
supported. Prefer `--agent-bin` pointing directly at a native executable or
npm JavaScript launcher rather than a shell alias. Image-provided Python and
tmux are preferred; the launcher's Python environment and bundled local tmux
dependencies provide fallbacks. Startup checks report missing or ABI-
incompatible tools without silently installing or updating them.

Full permissions are applied only to the sandbox invocation: Codex bypasses
approval/sandbox prompts, Claude bypasses permission prompts, and OpenCode
uses an allow-all runtime setting. Use `--permissions agent` to retain the
agent's normal approval settings. These switches do not grant root or
Kubernetes administration privileges.

Agent state is persistent, with agent-specific isolation:

- Codex: a filtered copy of `CODEX_HOME` (normally `~/.codex`), mounted at
  `/opt/falcon-agent-state`. Config, file-based login, profiles, instructions,
  and command history are seeded from the host. `sessions` and
  `archived_sessions` are mounted directly read-write, so existing and new
  conversation transcripts are shared without copying or size limits. Do not
  edit/resume the same conversation concurrently from host and sandbox.
  The broker
  synchronizes approved settings/login/history files in both directions every
  ten seconds. Concurrent config changes preserve both versions and report a
  conflict in the broker log; append-only history is merged. Deletions are not
  propagated. Skills/prompts/rules/agent definitions and regular plugin files
  are copied at launch without write-back; external skills are read-only. Daemon sockets,
  temporary files, logs, and SQLite databases remain container-local.
- Claude: `CLAUDE_CONFIG_DIR` (normally `~/.claude`). Legacy `~/.claude.json`
  is mirrored inside the session and synchronized by the broker. Concurrent
  host changes win conflicting top-level keys and produce a warning.
- OpenCode: its agent-specific XDG config, data, and state directories.

Only the selected agent's directories are exposed. The rest of home is not
mounted. Tool installations below writable state directories are protected
read-only. Updating or changing synchronized/shared settings can affect subsequent local
agent runs, and the agent can read its provider credentials. This is an
intentional exception to workspace-only host writes—not hostile-code VM
isolation. Do not place Kubernetes credentials or privileged host connectors
inside the allowed workspace/data/agent-state directories.

Codex copies are kept in a private `.falcon-sandbox-state/` directory beside
the original Codex home, outside the workspace and child mounts. They remain
available across Pod replacement and are retained after stop. Only regular files are copied;
symlinks, sockets, hard links, and unknown runtime files are excluded. Sync is
bounded to 20,000 inventory entries and 128 MiB per copied file; directly
mounted conversation transcripts are excluded from both limits. File-based login is
required to reuse a login; host OS keyring credentials are not exported.
SQLite is explicitly redirected to ephemeral scratch even when host config
sets `sqlite_home`.

See Codex's official [state-location documentation](https://learn.chatgpt.com/docs/config-file/environment-variables)
and [credential-storage documentation](https://learn.chatgpt.com/docs/auth)
for the underlying configuration and login conventions.

Claude/OpenCode still use direct shared state with protective runtime masks.
Some shared filesystem/runtime combinations reject overlays on private `0700`
directories. Falcon does not relax permissions or expose daemon sockets to
work around that limitation. Codex's filtered copy avoids these overlays.

## How child Jobs are restricted

Inside the sandbox, `falcon` is a restricted, dependency-free client:

```console
falcon h100x2 --name train -- python train.py
falcon -c 8 -m 16Gi -- python preprocess.py
falcon jobs
falcon get JOB
falcon logs JOB --no-follow --tail 100
falcon metrics JOB --interval 10
falcon kill JOB
```

Child names are prefixed with the session name; use the returned full name.
Logs are bounded to 2,000 lines per request and metrics intervals to 30
seconds. `--follow` polls the bounded log tail. Normal Falcon CLI/TUI commands
outside the sandbox are unchanged; the sandbox client does not implement
cluster-wide dashboard/resources, Coder, or raw kubectl operations.

Helper scripts are projected from an immutable ConfigMap, and the parent
session specification is projected from a Secret. These are owned by the
parent Job and garbage-collected when it is removed. They contain no
Kubernetes credentials or authoritative broker policy.

The client posts a typed request to a private, token-authenticated in-Pod
HTTP relay. A detached worker on the launch machine polls that relay through
loopback-only `kubectl port-forward`, validates the request against its
launch-time policy, and submits a reconstructed Job using local credentials.
It posts the result back through the relay. No public inbound port, broker
endpoint, or kubeconfig is exposed to the agent. The relay never receives
Kubernetes credentials and is not a trusted manifest source.

Every child gets the original workspace read-write, approved data read-only,
and read-only Falcon/Python runtime helpers. It receives only a session
capability—not provider credentials or writable agent state. Children can
submit further compute Jobs through the same broker but cannot launch nested
agent sandboxes, broaden mounts, select service accounts, enable privilege,
change namespace, or manage other sessions' Jobs. Requested environments must
already lie within the workspace or approved read-only data mounts; the
launch-time active environment is separately approved for read-write access.

Jobs run non-root with a read-only image filesystem, writable ephemeral
scratch, dropped capabilities, seccomp, disabled privilege escalation, and no
automounted Kubernetes service-account token. Existing namespace quotas still
apply. Submission request IDs and Job UIDs are stored on the launch machine
to prevent duplicate launches and name-reuse attacks.

## Broker availability and troubleshooting

The launch machine must remain running for new submissions, status/log
requests through the client, Codex state synchronization, and legacy Claude configuration synchronization.
Closing the attached terminal is safe. If that machine goes offline, existing
Jobs keep running; requests wait and eventually report a retry ID. Reconnect
on the original machine with `falcon sandbox resume NAME`. Broker state and
logs live under `$XDG_STATE_HOME/falcon/sandboxes/NAME` (normally
`~/.local/state/falcon/sandboxes/NAME`); this directory is never mounted into
the agent. The broker uses a frozen Falcon package, not the editable checkout.

Use `falcon sandbox --dry-run -- codex` to inspect the Job and Service without
creating state, credentials, or workloads. Add `--output json` for structured
output. If an agent is missing, specify its executable explicitly. Inspect
startup logs for incompatible binaries, unavailable shared paths, or image
dependencies. Installation updates take effect on the next sandbox launch.
