"""Fast, side-effect-free Bash and Zsh completion for Falcon."""

from __future__ import annotations

import shlex
from typing import Any, Dict, List, Optional

from .commands import job_names


BASE_COMMANDS = [
    "submit",
    "jobs",
    "get",
    "logs",
    "events",
    "delete",
    "attach",
    "top",
    "clean",
    "dashboard",
    "resources",
    "setup",
    "config",
    "completion",
]
JOB_COMMANDS = {"get", "logs", "events", "delete", "attach", "top"}

OPTIONS: Dict[str, List[str]] = {
    "submit": [
        "--gpu", "--gpus", "--cpu", "--memory", "--name", "--environment",
        "--image", "--env", "--shm-size", "--shm-percent", "--async",
        "--pin-node", "--max", "--dry-run", "--output", "--",
    ],
    "jobs": [
        "--output", "--limit", "--status", "--gpu", "--node", "--namespace",
    ],
    "get": ["--output", "--events", "--namespace"],
    "logs": ["--tail", "--follow", "--container", "--namespace"],
    "events": ["--output", "--limit", "--namespace"],
    "delete": ["--output", "--namespace"],
    "dashboard": ["--demo", "--job"],
    "resources": ["--output", "--node", "--gpu", "--limit", "--demo"],
    "setup": [
        "--force", "--non-interactive", "--no-shell", "--install-skills",
        "--skip-skills", "--uninstall-skills",
    ],
    "completion": ["bash", "zsh"],
}


def preset_tokens(config: Dict[str, Any]) -> List[str]:
    """Return bounded completion hints; arbitrary positive ``xN`` still parses."""
    values: List[str] = []
    for name in config.get("presets", {}):
        values.append(name)
        values.extend(f"{name}x{count}" for count in range(2, 9))
    return values


def candidates(
    kind: str, config: Dict[str, Any], command: str = ""
) -> List[str]:
    if kind == "commands":
        return BASE_COMMANDS + preset_tokens(config)
    if kind == "jobs":
        return job_names(config["cluster"]["namespace"])
    if kind == "options":
        if command in OPTIONS:
            return OPTIONS[command]
        if any(
            command == name
            or (
                command.startswith(name + "x")
                and command[len(name) + 1 :].isdigit()
            )
            for name in config.get("presets", {})
        ):
            return OPTIONS["submit"]
    return []


def _words(values: List[str]) -> str:
    return " ".join(shlex.quote(value) for value in values)


def shell_script(
    shell: str, launcher: str = "", config: Optional[Dict[str, Any]] = None
) -> str:
    """Generate completion without wrapping or replacing the Falcon executable."""
    del launcher  # Retained in the signature for callers of older Falcon builds.
    config = config or {"presets": {}, "cluster": {"namespace": "default"}}
    commands = _words(BASE_COMMANDS + preset_tokens(config))
    presets = _words(list(config.get("presets", {})))
    namespace = shlex.quote(str(config["cluster"]["namespace"]))
    option_cases = "\n".join(
        f"      {shlex.quote(name)}) values=({_words(values)}) ;;"
        for name, values in OPTIONS.items()
    )
    if shell == "zsh":
        return f"""# Falcon completion (the executable remains a normal PATH command)
typeset -ga _falcon_commands=({commands})
typeset -ga _falcon_presets=({presets})
typeset -ga _falcon_job_cache=()
typeset -gi _falcon_job_cache_time=-2
zmodload zsh/datetime 2>/dev/null
_falcon_refresh_jobs() {{
  local now=${{EPOCHSECONDS:-$SECONDS}}
  if (( now - _falcon_job_cache_time >= 2 )); then
    local -a raw
    raw=("${{(@f)$(command kubectl get jobs.batch -n {namespace} -o name 2>/dev/null)}}")
    _falcon_job_cache=("${{raw[@]#job.batch/}}")
    _falcon_job_cache_time=$now
  fi
}}
_falcon_native() {{
  local subject="${{words[2]}}"
  local -a values
  if (( CURRENT == 2 )); then
    values=("${{_falcon_commands[@]}}")
  elif [[ "$subject" == get || "$subject" == logs || "$subject" == events || "$subject" == delete || "$subject" == attach || "$subject" == top ]]; then
    _falcon_refresh_jobs
    values=("${{_falcon_job_cache[@]}}")
  else
    case "$subject" in
{option_cases}
      *) values=() ;;
    esac
  fi
  compadd -- $values
}}
compdef _falcon_native falcon
"""
    if shell == "bash":
        bash_cases = "\n".join(
            f"      {name}) values=({_words(values)}) ;;"
            for name, values in OPTIONS.items()
        )
        return f"""# Falcon completion (the executable remains a normal PATH command)
_falcon_commands=({commands})
_falcon_job_cache=()
_falcon_job_cache_time=-2
_falcon_refresh_jobs() {{
  local now=$SECONDS value
  if (( now - _falcon_job_cache_time >= 2 )); then
    _falcon_job_cache=()
    while IFS= read -r value; do
      [[ -n "$value" ]] && _falcon_job_cache+=("${{value#job.batch/}}")
    done < <(command kubectl get jobs.batch -n {namespace} -o name 2>/dev/null)
    _falcon_job_cache_time=$now
  fi
}}
_falcon_native() {{
  local cur="${{COMP_WORDS[COMP_CWORD]}}" subject="${{COMP_WORDS[1]}}"
  local -a values=()
  if [[ $COMP_CWORD -eq 1 ]]; then
    values=("${{_falcon_commands[@]}}")
  elif [[ "$subject" =~ ^(get|logs|events|delete|attach|top)$ ]]; then
    _falcon_refresh_jobs
    values=("${{_falcon_job_cache[@]}}")
  else
    case "$subject" in
{bash_cases}
      *) values=() ;;
    esac
  fi
  COMPREPLY=( $(compgen -W "${{values[*]}}" -- "$cur") )
}}
complete -F _falcon_native falcon
"""
    raise ValueError("completion shell must be zsh or bash")
