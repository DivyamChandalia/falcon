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

The skill deliberately teaches only the normal user workflow:

```console
falcon h100x2 --name run -- python train.py
falcon logs run --no-follow --tail 100
falcon metrics run --interval 60
falcon kill run
```

Launches are detached by default. After launch, agents report the Job name and
return unless observation was explicitly requested; they must not add
`-f`/`--follow` themselves. Metrics always emits JSON. The skill records the
eviction averages—90% for H100 and 30% for A6000/2080Ti—and directs all other
specific inspection to one targeted `kubectl` command.

Managed installs are idempotent. Falcon updates unchanged copies, removes
obsolete Falcon-owned skill files, and never overwrites user modifications.
