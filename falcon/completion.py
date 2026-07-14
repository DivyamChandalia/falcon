"""Dynamic zsh/bash completions for Falcon."""

from __future__ import annotations

import json
import os
import shlex
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from .commands import job_names
from .resources import canonical_gpu, fetch_nodes


BASE_COMMANDS = [
    "setup", "dashboard", "dash", "logs", "attach", "top", "delete", "kill", "clean",
    "config", "shell-init", "completion",
]
LEGACY_SUBMISSION_OPTIONS = ["-j", "--job", "-g", "--gpu-type", "-n", "--num-gpus"]
JOB_COMMANDS = {"logs", "attach", "top", "delete", "kill"}
PRESET_CACHE_TTL_SECONDS = 300


def _preset_cache_path() -> Path:
    return Path(os.environ.get("XDG_CACHE_HOME", "~/.cache")).expanduser() / "falcon" / "preset-capacities.json"


def _preset_signature(config: Dict[str, Any]) -> str:
    return json.dumps(
        {
            "url": config["cluster"].get("kube_state_metrics_url"),
            "presets": {
                name: preset.get("gpu_type") for name, preset in config["presets"].items()
            },
        },
        sort_keys=True,
    )


def preset_tokens(config: Dict[str, Any]) -> List[str]:
    cache_path = _preset_cache_path()
    signature = _preset_signature(config)
    try:
        if time.time() - cache_path.stat().st_mtime < PRESET_CACHE_TTL_SECONDS:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            if cached.get("signature") == signature and isinstance(cached.get("tokens"), list):
                return [str(value) for value in cached["tokens"]]
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        pass
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
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = cache_path.with_name(f"{cache_path.name}.{os.getpid()}.tmp")
        temporary.write_text(json.dumps({"signature": signature, "tokens": tokens}), encoding="utf-8")
        temporary.replace(cache_path)
    except OSError:
        pass
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


def _shell_words(values: List[str]) -> str:
    return " ".join(shlex.quote(value) for value in values)


def shell_script(shell: str, launcher: str = "", config: Optional[Dict[str, Any]] = None) -> str:
    """Generate completion with static values embedded for zero-startup Tab presses."""
    executable = shlex.quote(launcher or str(Path.home() / ".local" / "bin" / "falcon"))
    config = config or {"presets": {}, "cluster": {"namespace": "default"}}
    preset_names = list(config["presets"])
    command_values = BASE_COMMANDS + preset_tokens(config) + LEGACY_SUBMISSION_OPTIONS
    namespace = shlex.quote(str(config["cluster"]["namespace"]))
    replacements = {
        "__FALCON__": executable,
        "__COMMANDS__": _shell_words(command_values),
        "__PRESETS__": _shell_words(preset_names),
        "__PRESET_OPTIONS__": _shell_words(candidates("options", config, preset_names[0]) if preset_names else []),
        "__DASHBOARD_OPTIONS__": _shell_words(candidates("options", config, "dashboard")),
        "__SETUP_OPTIONS__": _shell_words(candidates("options", config, "setup")),
        "__SHELL_OPTIONS__": _shell_words(candidates("options", config, "shell-init")),
        "__NAMESPACE__": namespace,
    }
    if shell == "zsh":
        script = r'''# Native Falcon command and cached native completion
unfunction _falcon 2>/dev/null
function falcon { __FALCON__ "$@"; }
typeset -ga _falcon_commands=(__COMMANDS__)
typeset -ga _falcon_presets=(__PRESETS__)
typeset -ga _falcon_preset_options=(__PRESET_OPTIONS__)
typeset -ga _falcon_dashboard_options=(__DASHBOARD_OPTIONS__)
typeset -ga _falcon_setup_options=(__SETUP_OPTIONS__)
typeset -ga _falcon_shell_options=(__SHELL_OPTIONS__)
typeset -ga _falcon_job_cache=()
typeset -gi _falcon_job_cache_time=-2
zmodload zsh/datetime 2>/dev/null
_falcon_refresh_jobs() {
  local now=${EPOCHSECONDS:-$SECONDS}
  if (( now - _falcon_job_cache_time >= 2 )); then
    local -a raw
    raw=("${(@f)$(command kubectl get jobs.batch -n __NAMESPACE__ -o name 2>/dev/null)}")
    _falcon_job_cache=("${raw[@]#job.batch/}")
    _falcon_job_cache_time=$now
  fi
}
_falcon_native() {
  local dash_index=${words[(I)--]}
  if (( dash_index > 0 && dash_index < CURRENT )); then
    shift $dash_index words
    (( CURRENT -= dash_index ))
    _normal
    return
  fi
  local -a values
  local subject="${words[2]}"
  if (( CURRENT == 2 )); then
    values=("${_falcon_commands[@]}")
  elif (( CURRENT >= 3 )) && [[ "$subject" == "logs" || "$subject" == "attach" || "$subject" == "top" || "$subject" == "delete" || "$subject" == "kill" ]]; then
    _falcon_refresh_jobs
    values=("${_falcon_job_cache[@]}")
  elif [[ "$subject" == "dashboard" || "$subject" == "dash" ]]; then
    values=("${_falcon_dashboard_options[@]}")
  elif [[ "$subject" == "setup" ]]; then
    values=("${_falcon_setup_options[@]}")
  elif [[ "$subject" == "completion" || "$subject" == "shell-init" ]]; then
    values=("${_falcon_shell_options[@]}")
  else
    local preset
    for preset in "${_falcon_presets[@]}"; do
      if [[ "$subject" == "$preset" || "$subject" == ${preset}x<-> ]]; then
        values=("${_falcon_preset_options[@]}")
        break
      fi
    done
  fi
  compadd -- $values
}
compdef _falcon_native falcon'''
        for marker, value in replacements.items():
            script = script.replace(marker, value)
        return script
    if shell == "bash":
        script = r'''# Native Falcon command and cached native completion
falcon() { __FALCON__ "$@"; }
_falcon_commands=(__COMMANDS__)
_falcon_presets=(__PRESETS__)
_falcon_preset_options=(__PRESET_OPTIONS__)
_falcon_dashboard_options=(__DASHBOARD_OPTIONS__)
_falcon_setup_options=(__SETUP_OPTIONS__)
_falcon_shell_options=(__SHELL_OPTIONS__)
_falcon_job_cache=()
_falcon_job_cache_time=-2
_falcon_refresh_jobs() {
  local now=$SECONDS
  if (( now - _falcon_job_cache_time >= 2 )); then
    local value
    _falcon_job_cache=()
    while IFS= read -r value; do
      [[ -n "$value" ]] && _falcon_job_cache+=("${value#job.batch/}")
    done < <(command kubectl get jobs.batch -n __NAMESPACE__ -o name 2>/dev/null)
    _falcon_job_cache_time=$now
  fi
}
_falcon_native() {
  local cur="${COMP_WORDS[COMP_CWORD]}"
  local subject="${COMP_WORDS[1]}"
  local -a values=()
  if [[ $COMP_CWORD -eq 1 ]]; then
    values=("${_falcon_commands[@]}")
  elif [[ "$subject" =~ ^(logs|attach|top|delete|kill)$ && $COMP_CWORD -ge 2 ]]; then
    _falcon_refresh_jobs
    values=("${_falcon_job_cache[@]}")
  elif [[ "$subject" =~ ^(dashboard|dash)$ ]]; then
    values=("${_falcon_dashboard_options[@]}")
  elif [[ "$subject" == setup ]]; then
    values=("${_falcon_setup_options[@]}")
  elif [[ "$subject" =~ ^(completion|shell-init)$ ]]; then
    values=("${_falcon_shell_options[@]}")
  else
    local preset suffix
    for preset in "${_falcon_presets[@]}"; do
      suffix="${subject#${preset}x}"
      if [[ "$subject" == "$preset" || ( "$subject" == "${preset}x"* && "$suffix" =~ ^[0-9]+$ ) ]]; then
        values=("${_falcon_preset_options[@]}")
        break
      fi
    done
  fi
  COMPREPLY=( $(compgen -W "${values[*]}" -- "$cur") )
}
complete -F _falcon_native falcon'''
        for marker, value in replacements.items():
            script = script.replace(marker, value)
        return script
    raise ValueError("completion shell must be zsh or bash")
