"""Fast, side-effect-free Bash and Zsh completion for Falcon."""

from __future__ import annotations

import re
import shlex
from typing import Any, Dict, List, Optional

from .commands import job_names
from .config import gpu_preset_max_count

BASE_COMMANDS = [
    "jobs",
    "get",
    "logs",
    "events",
    "kill",
    "attach",
    "top",
    "metrics",
    "clean",
    "dashboard",
    "resources",
    "coder",
    "setup",
    "config",
    "completion",
]
PRIORITY_COMMANDS = ["logs", "top"]
TOP_LEVEL_LAUNCH_OPTIONS = ["--cpu", "--memory", "--gpus"]
COMMAND_ALIASES = {
    "j": "jobs",
    "g": "get",
    "e": "events",
    "l": "logs",
    "a": "attach",
    "t": "top",
    "m": "metrics",
    "k": "kill",
    "c": "clean",
    "d": "dashboard",
    "r": "resources",
    "s": "setup",
}
JOB_COMMANDS = {"get", "logs", "events", "kill", "attach", "top", "metrics"}

LAUNCH_OPTIONS = [
    "--gpus", "--cpu", "--memory", "--name", "--environment",
    "--image", "--env", "--shm-size", "--shm-percent", "-f", "--follow",
    "--pin-node", "--max", "--dry-run", "--output", "--namespace", "--",
]

OPTIONS: Dict[str, List[str]] = {
    "jobs": [
        "--output", "--limit", "--status", "--gpu", "--node", "--namespace",
    ],
    "get": ["--output", "--namespace"],
    "logs": [
        "--tail", "--follow", "--no-follow", "--container", "--namespace",
        "--output",
    ],
    "events": ["-f", "--follow", "--output", "--limit", "--namespace"],
    "top": ["--namespace"],
    "metrics": ["--interval", "--namespace"],
    "kill": ["--output", "--namespace"],
    "dashboard": ["--demo", "--job", "--color"],
    "resources": [
        "--output", "--node", "--gpu", "--limit", "--consumer-limit", "--demo",
        "--namespace", "--color",
    ],
    "coder": [
        "-c", "--cpu", "-m", "--memory", "-j", "--name", "--template", "--url",
        "--parameter", "--access", "--timeout",
    ],
    "setup": [
        "--force", "--non-interactive", "--no-shell", "--install-skills",
        "--skip-skills", "--uninstall-skills",
    ],
    "completion": ["bash", "zsh"],
}


def preset_tokens(config: Dict[str, Any]) -> List[str]:
    """Return only base presets; GPU counts are completed as the next word."""

    return list(config.get("presets", {}))


def counted_preset_tokens(
    config: Dict[str, Any], preset: str = ""
) -> List[str]:
    return [
        f"{name}x{count}"
        for name, preset_config in config.get("presets", {}).items()
        if not preset or name == preset
        for count in range(2, gpu_preset_max_count(preset_config) + 1)
    ]


def candidates(
    kind: str, config: Dict[str, Any], command: str = ""
) -> List[str]:
    if kind == "commands":
        remaining = [
            value for value in BASE_COMMANDS
            if value not in PRIORITY_COMMANDS
        ]
        return (
            preset_tokens(config)
            + PRIORITY_COMMANDS
            + remaining
        )
    if kind == "counts":
        return counted_preset_tokens(config, command)
    if kind == "jobs":
        return job_names(config["cluster"]["namespace"])
    if kind == "options":
        command = COMMAND_ALIASES.get(command, command)
        if command == "coder":
            return (
                preset_tokens(config)
                + counted_preset_tokens(config)
                + OPTIONS[command]
            )
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
            return LAUNCH_OPTIONS
    return []


def _words(values: List[str]) -> str:
    return " ".join(shlex.quote(value) for value in values)


def shell_script(
    shell: str, launcher: str = "", config: Optional[Dict[str, Any]] = None
) -> str:
    """Generate completion without wrapping or replacing the Falcon executable."""
    del launcher  # Retained in the signature for callers of older Falcon builds.
    config = config or {"presets": {}, "cluster": {"namespace": "default"}}
    ordered_commands = (
        preset_tokens(config)
        + PRIORITY_COMMANDS
        + [
            value for value in BASE_COMMANDS
            if value not in PRIORITY_COMMANDS
        ]
    )
    commands = _words(ordered_commands)
    top_level_options = _words(TOP_LEVEL_LAUNCH_OPTIONS)
    presets = _words(list(config.get("presets", {})))
    counted_presets = _words(counted_preset_tokens(config))
    launch_options = _words(LAUNCH_OPTIONS)
    namespace = shlex.quote(str(config["cluster"]["namespace"]))
    preset_names = [str(name) for name in config.get("presets", {})]
    zsh_base_preset_pattern = "|".join(
        map(shlex.quote, preset_names)
    ) or "__no_falcon_presets__"
    zsh_counted_preset_pattern = "|".join(
        f"{shlex.quote(name)}x*" for name in preset_names
    ) or "__no_falcon_counted_presets__"
    bash_preset_pattern = "|".join(re.escape(name) for name in preset_names) or "(?!)"
    alias_cases = "\n".join(
        f"    {alias}) subject={canonical} ;;"
        for alias, canonical in COMMAND_ALIASES.items()
    )
    option_cases = "\n".join(
        f"      {shlex.quote(name)}) values=({_words(values)}) ;;"
        for name, values in OPTIONS.items()
    )
    if shell == "zsh":
        return f"""# Falcon completion (the executable remains a normal PATH command)
typeset -ga _falcon_commands=({commands})
typeset -ga _falcon_top_level_options=({top_level_options})
typeset -ga _falcon_presets=({presets})
typeset -ga _falcon_counted_presets=({counted_presets})
typeset -ga _falcon_launch_options=({launch_options})
_falcon_native() {{
  local subject="${{words[2]}}"
  local current="${{words[CURRENT]}}"
  local -a values
  compstate[list]="${{compstate[list]}} rows"
  case "$subject" in
{alias_cases}
  esac
  if (( CURRENT == 2 )); then
    case "$current" in
      {zsh_base_preset_pattern})
        values=("${{(@M)_falcon_counted_presets:#${{current}}x*}}")
        compadd -V falcon-presets -S '' -- $values
        return
        ;;
      {zsh_counted_preset_pattern})
        values=("${{_falcon_counted_presets[@]}}")
        compadd -V falcon-presets -- $values
        return
        ;;
    esac
    values=("${{_falcon_commands[@]}}" "${{_falcon_top_level_options[@]}}")
    local preset
    for preset in "${{_falcon_presets[@]}}"; do
      if [[ -n "$current" && "$preset" == "$current"* ]]; then
        compadd -V falcon-top -S '' -- $values
        return
      fi
    done
  elif [[ "$subject" == coder ]]; then
    values=()
    local value
    while IFS= read -r value; do
      [[ -n "$value" && "$value" != "<none>" ]] && values+=("$value")
    done < <(command kubectl get jobs.batch -n {namespace} --selector app.kubernetes.io/managed-by=coder --output 'custom-columns=WORKSPACE:.metadata.labels.coder\\.workspace' --no-headers 2>/dev/null)
  elif (( CURRENT == 3 )) && [[ "$subject" == get || "$subject" == logs || "$subject" == events || "$subject" == kill || "$subject" == attach || "$subject" == top || "$subject" == metrics ]]; then
    values=("${{(@f)$(command kubectl get jobs.batch -n {namespace} -o name 2>/dev/null)}}")
    values=("${{values[@]#job.batch/}}")
  elif [[ "$subject" == -c || "$subject" == --cpu || "$subject" == -m || "$subject" == --memory || "$subject" == --gpu || "$subject" == --gpus || "$subject" == -n ]]; then
    values=("${{_falcon_launch_options[@]}}")
  else
    case "$subject" in
      {zsh_base_preset_pattern}) values=("${{_falcon_launch_options[@]}}") ;;
      {zsh_counted_preset_pattern}) values=("${{_falcon_launch_options[@]}}") ;;
{option_cases}
      *) values=() ;;
    esac
  fi
  compadd -V falcon-values -- $values
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
_falcon_top_level_options=({top_level_options})
_falcon_counted_presets=({counted_presets})
_falcon_launch_options=({launch_options})
_falcon_native() {{
  local cur="${{COMP_WORDS[COMP_CWORD]}}" subject="${{COMP_WORDS[1]}}"
  local -a values=()
  case "$subject" in
{alias_cases}
  esac
  if [[ $COMP_CWORD -eq 1 ]]; then
    if [[ "$cur" =~ ^({bash_preset_pattern})$ ]]; then
      values=("${{_falcon_counted_presets[@]}}")
      compopt -o nospace 2>/dev/null || true
    elif [[ "$cur" =~ ^({bash_preset_pattern})x[0-9]*$ ]]; then
      values=("${{_falcon_counted_presets[@]}}")
    else
      values=("${{_falcon_commands[@]}}" "${{_falcon_top_level_options[@]}}")
      local preset
      for preset in {' '.join(shlex.quote(name) for name in preset_names)}; do
        if [[ -n "$cur" && "$preset" == "$cur"* ]]; then
          compopt -o nospace 2>/dev/null || true
          break
        fi
      done
    fi
  elif [[ "$subject" == coder ]]; then
    values=()
    local value
    while IFS= read -r value; do
      [[ -n "$value" && "$value" != "<none>" ]] && values+=("$value")
    done < <(command kubectl get jobs.batch -n {namespace} --selector app.kubernetes.io/managed-by=coder --output 'custom-columns=WORKSPACE:.metadata.labels.coder\\.workspace' --no-headers 2>/dev/null)
  elif [[ $COMP_CWORD -eq 2 ]] && [[ "$subject" =~ ^(get|logs|events|kill|attach|top|metrics)$ ]]; then
    values=()
    local value
    while IFS= read -r value; do
      [[ -n "$value" ]] && values+=("${{value#job.batch/}}")
    done < <(command kubectl get jobs.batch -n {namespace} -o name 2>/dev/null)
  elif [[ "$subject" =~ ^(-c|--cpu|-m|--memory|--gpu|--gpus|-n)$ ]]; then
    values=("${{_falcon_launch_options[@]}}")
  elif [[ "$subject" =~ ^({bash_preset_pattern})(x[1-9][0-9]*)?$ ]]; then
    values=("${{_falcon_launch_options[@]}}")
  else
    case "$subject" in
{bash_cases}
      *) values=() ;;
    esac
  fi
  compopt -o nosort 2>/dev/null || true
  COMPREPLY=( $(compgen -W "${{values[*]}}" -- "$cur") )
}}
complete -F _falcon_native falcon
"""
    raise ValueError("completion shell must be zsh or bash")
