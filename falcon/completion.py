"""Dynamic zsh/bash completions for Falcon."""

from __future__ import annotations

import shlex
from pathlib import Path
from typing import Any, Dict, List

from .commands import job_names
from .resources import canonical_gpu, fetch_nodes


BASE_COMMANDS = [
    "setup", "dashboard", "dash", "logs", "attach", "top", "delete", "kill", "clean",
    "config", "shell-init", "completion",
]
LEGACY_SUBMISSION_OPTIONS = ["-j", "--job", "-g", "--gpu-type", "-n", "--num-gpus"]
JOB_COMMANDS = {"logs", "attach", "top", "delete", "kill"}


def preset_tokens(config: Dict[str, Any]) -> List[str]:
    maximums = {name: 1 for name in config["presets"]}
    try:
        nodes = fetch_nodes(config["cluster"]["kube_state_metrics_url"], timeout=3)
        for name, preset in config["presets"].items():
            counts = [
                node.gpu_total for node in nodes
                if canonical_gpu(node.gpu_product) == canonical_gpu(preset["gpu_type"])
            ]
            if counts:
                maximums[name] = max(counts)
    except Exception:
        pass
    tokens = []
    for name in config["presets"]:
        tokens.append(name)
        tokens.extend(f"{name}x{count}" for count in range(2, maximums[name] + 1))
    return tokens


def candidates(kind: str, config: Dict[str, Any], command: str = "") -> List[str]:
    if kind == "commands":
        return BASE_COMMANDS + preset_tokens(config) + LEGACY_SUBMISSION_OPTIONS
    if kind == "jobs":
        return job_names(config["cluster"]["namespace"])
    if kind == "options":
        if command in {"dashboard", "dash"}:
            return ["--once", "--json", "--job", "--samples", "--interval"]
        if command in JOB_COMMANDS or command == "clean":
            return []
        if command == "setup":
            return ["--force", "--non-interactive", "--no-shell"]
        if command in {"completion", "shell-init"}:
            return ["zsh", "bash"]
        is_preset = any(
            command == name or (
                command.startswith(name + "x") and command[len(name) + 1:].isdigit()
            )
            for name in config["presets"]
        )
        if is_preset:
            return [
                "--cpu", "--memory", "--shm-size", "--shm-percent", "--job", "--async",
                "--max", "--pin-node", "--dry-run", "--explain", "--jet-arg", "--",
            ]
    return []


def shell_script(shell: str, launcher: str = "") -> str:
    executable = shlex.quote(launcher or str(Path.home() / ".local" / "bin" / "falcon"))
    if shell == "zsh":
        script = r'''# Native Falcon command and dynamic completion
unfunction _falcon 2>/dev/null
function falcon { __FALCON__ "$@"; }
_falcon_native() {
  local dash_index=${words[(I)--]}
  if (( dash_index > 0 && dash_index < CURRENT )); then
    shift $dash_index words
    (( CURRENT -= dash_index ))
    _normal
    return
  fi
  local -a values
  if (( CURRENT == 2 )); then
    values=("${(@f)$(__FALCON__ _complete commands 2>/dev/null)}")
  elif [[ "${words[CURRENT-1]}" == "--job" ]]; then
    values=("${(@f)$(__FALCON__ _complete jobs 2>/dev/null)}")
  elif (( CURRENT == 3 )) && [[ "${words[2]}" == "logs" || "${words[2]}" == "attach" || "${words[2]}" == "top" || "${words[2]}" == "delete" || "${words[2]}" == "kill" ]]; then
    values=("${(@f)$(__FALCON__ _complete jobs 2>/dev/null)}")
  else
    values=("${(@f)$(__FALCON__ _complete options "${words[2]}" 2>/dev/null)}")
  fi
  compadd -- $values
}
compdef _falcon_native falcon'''
        return script.replace("__FALCON__", executable)
    if shell == "bash":
        script = r'''# Native Falcon command and dynamic completion
falcon() { __FALCON__ "$@"; }
_falcon_native() {
  local cur="${COMP_WORDS[COMP_CWORD]}"
  local values
  if [[ $COMP_CWORD -eq 1 ]]; then
    values="$(__FALCON__ _complete commands 2>/dev/null)"
  elif [[ "${COMP_WORDS[COMP_CWORD-1]}" == "--job" ]]; then
    values="$(__FALCON__ _complete jobs 2>/dev/null)"
  elif [[ "${COMP_WORDS[1]}" =~ ^(logs|attach|top|delete|kill)$ && $COMP_CWORD -eq 2 ]]; then
    values="$(__FALCON__ _complete jobs 2>/dev/null)"
  else
    values="$(__FALCON__ _complete options "${COMP_WORDS[1]}" 2>/dev/null)"
  fi
  COMPREPLY=( $(compgen -W "$values" -- "$cur") )
}
complete -F _falcon_native falcon'''
        return script.replace("__FALCON__", executable)
    raise ValueError("completion shell must be zsh or bash")
