"""Fast, side-effect-free Bash and Zsh completion for Falcon."""

from __future__ import annotations

import re
import shlex
from typing import Any, Dict, List, Optional

from .commands import job_names
from .config import gpu_preset_max_count

BASE_COMMANDS = [
    "sandbox",
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
    "update",
    "mcp",
]
PRIORITY_COMMANDS = ["logs", "top"]
TOP_LEVEL_LAUNCH_OPTIONS = ["--cpu", "--memory", "--gpus"]
JOB_COMMANDS = {"get", "logs", "events", "kill", "attach", "top", "metrics"}


def resolve_command(token: str) -> Optional[str]:
    """Resolve every command through the same unambiguous-prefix rule."""
    if token in BASE_COMMANDS:
        return token
    matches = [name for name in BASE_COMMANDS if token and name.startswith(token)]
    if len(matches) > 1:
        raise ValueError(f"Ambiguous command {token!r}: {', '.join(matches)}. Type more letters or press Tab.")
    return matches[0] if matches else None


# Shared by Bash and Zsh, with no Python startup or cluster request. Offer
# prefixes first, then at most two edits against full names or their prefixes.
_FUZZY_COMPLETION = r'''_falcon_suggest() {
  command awk -v word="$1" 'BEGIN {
    count = split(ARGV[1], names, " "); delete ARGV[1]; found = 0;
    for (k=1; k<=count; k++) if (index(names[k], word)==1) {
      print names[k]; found=1;
    }
    if (found || length(word)<4 || substr(word,1,1)=="-") exit;
    best=3;
    for (k=1; k<=count; k++) {
      name=names[k]; n=length(word); m=length(name);
      for (i=0; i<=n; i++) d[i,0]=i;
      for (j=0; j<=m; j++) d[0,j]=j;
      for (i=1; i<=n; i++) for (j=1; j<=m; j++) {
        v=d[i-1,j]+1;
        if (d[i,j-1]+1<v) v=d[i,j-1]+1;
        cost=(substr(word,i,1)!=substr(name,j,1));
        if (d[i-1,j-1]+cost<v) v=d[i-1,j-1]+cost;
        if (i>1 && j>1 && substr(word,i,1)==substr(name,j-1,1) &&
            substr(word,i-1,1)==substr(name,j,1) && d[i-2,j-2]+1<v) v=d[i-2,j-2]+1;
        d[i,j]=v;
      }
      distance[k]=d[n,m];
      for (j=3; j<m; j++) if (j>=n-2 && d[n,j]<distance[k]) distance[k]=d[n,j];
      if (distance[k]<best) best=distance[k];
    }
    for (k=1; k<=count; k++) if (distance[k]==best && best<=2) print names[k];
  }' "${_falcon_canonical_commands[*]}"
}
'''

LAUNCH_OPTIONS = [
    "--gpus", "--cpu", "--memory", "--name", "--environment",
    "--image", "--env", "--shm-size", "--shm-percent", "-f", "--follow",
    "--pin-node", "--max", "--dry-run", "--output", "--namespace", "--",
]

OPTIONS: Dict[str, List[str]] = {
    "sandbox": [*LAUNCH_OPTIONS, "--read-only", "--agent-bin", "--permissions", "--detach", "--auto-continue", "status", "resume", "stop"],
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
    "kill": ["--output", "--namespace", "--dry-run"],
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
        "--skip-skills", "--uninstall-skills", "--configure-mcp", "--skip-mcp",
        "--mcp-dry-run", "--skill-mode", "mcp", "legacy",
    ],
    "completion": ["bash", "zsh"],
    "update": ["--check"],
    "mcp": ["serve", "config", "--agent", "codex", "claude", "opencode", "--state-dir"],
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
        try:
            command = resolve_command(command) or command
        except ValueError:
            return []
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
    subject_aliases = {}
    for name in BASE_COMMANDS:
        for length in range(1, len(name)):
            prefix = name[:length]
            if prefix not in subject_aliases and prefix not in preset_names and sum(value.startswith(prefix) for value in BASE_COMMANDS) == 1:
                subject_aliases[prefix] = name
    alias_cases = "\n".join(
        f"    {alias}) subject={canonical} ;;"
        for alias, canonical in subject_aliases.items()
    )
    option_cases = "\n".join(
        f"      {shlex.quote(name)}) values=({_words(values)}) ;;"
        for name, values in OPTIONS.items()
    )
    if shell == "zsh":
        return f"""# Falcon completion (the executable remains a normal PATH command)
typeset -ga _falcon_commands=({commands})
typeset -ga _falcon_canonical_commands=({_words(BASE_COMMANDS)})
{_FUZZY_COMPLETION}
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
  if [[ "$subject" == sandbox ]]; then
    local previous="${{words[CURRENT-1]}}" index
    for (( index=3; index<CURRENT; index++ )); do
      if [[ "${{words[index]}}" == -- ]]; then
        _normal
        return
      fi
    done
    case "$previous" in
      --read-only|--environment) _files -/; return ;;
      --agent-bin) _files; return ;;
      --permissions) compadd -- full agent; return ;;
    esac
    if (( CURRENT == 4 )) && [[ "${{words[3]}}" == status || "${{words[3]}}" == resume || "${{words[3]}}" == stop ]]; then
      local name directory
      while IFS= read -r name; do
        name="${{name#job.batch/}}"
        directory="${{XDG_STATE_HOME:-$HOME/.local/state}}/falcon/sandboxes/$name"
        [[ -f "$directory/session.json" && ! -f "$directory/stopped" ]] && values+=("$name")
      done < <(command kubectl get jobs.batch -n {namespace} --selector falcon.dev/sandbox -o name --request-timeout=2s 2>/dev/null)
      compadd -- $values
      return
    fi
  fi
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
    if [[ -n "$current" && "$current" != -* ]]; then
      local -a suggestions
      suggestions=("${{(@f)$(_falcon_suggest "$current")}}")
      if (( ${{#suggestions}} )) && [[ -n "${{suggestions[1]}}" ]]; then
        compadd -U -V falcon-commands -- $suggestions
        return
      fi
    fi
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
_falcon_canonical_commands=({_words(BASE_COMMANDS)})
{_FUZZY_COMPLETION}
_falcon_top_level_options=({top_level_options})
_falcon_counted_presets=({counted_presets})
_falcon_launch_options=({launch_options})
_falcon_native() {{
  local cur="${{COMP_WORDS[COMP_CWORD]}}" subject="${{COMP_WORDS[1]}}"
  local -a values=()
  case "$subject" in
{alias_cases}
  esac
  COMPREPLY=()
  if [[ "$subject" == sandbox ]]; then
    local previous="${{COMP_WORDS[COMP_CWORD-1]}}" word
    for word in "${{COMP_WORDS[@]:2:COMP_CWORD-2}}"; do
      if [[ "$word" == -- ]]; then
        compopt -o default -o bashdefault 2>/dev/null || true
        return
      fi
    done
    case "$previous" in
      --read-only|--environment|--agent-bin)
        local lookup="$cur" path mode=-d
        [[ "$previous" == --agent-bin ]] && mode=-f
        [[ "$lookup" == '~/'* ]] && lookup="$HOME/${{lookup:2}}"
        while IFS= read -r path; do
          [[ -d "$path" ]] && path="${{path%/}}/"
          [[ "$cur" == '~/'* ]] && path="~/${{path#"$HOME/"}}"
          COMPREPLY+=("$path")
        done < <(compgen "$mode" -- "$lookup")
        compopt -o filenames -o nospace 2>/dev/null || true
        return ;;
      --permissions) values=(full agent) ;;
    esac
    if [[ "$previous" == --permissions ]]; then
      COMPREPLY=( $(compgen -W "${{values[*]}}" -- "$cur") )
      return
    fi
    if [[ $COMP_CWORD -eq 3 && "${{COMP_WORDS[2]}}" =~ ^(status|resume|stop)$ ]]; then
      local name directory
      while IFS= read -r name; do
        name="${{name#job.batch/}}"
        directory="${{XDG_STATE_HOME:-$HOME/.local/state}}/falcon/sandboxes/$name"
        [[ -f "$directory/session.json" && ! -f "$directory/stopped" ]] || continue
        [[ "$name" == "$cur"* ]] && COMPREPLY+=("$name")
      done < <(command kubectl get jobs.batch -n {namespace} --selector falcon.dev/sandbox -o name --request-timeout=2s 2>/dev/null)
      return
    fi
  fi
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
  if [[ $COMP_CWORD -eq 1 && ${{#COMPREPLY[@]}} -eq 0 ]]; then
    local suggestion
    while IFS= read -r suggestion; do
      [[ -n "$suggestion" ]] && COMPREPLY+=("$suggestion")
    done < <(_falcon_suggest "$cur")
  fi
}}
complete -F _falcon_native falcon
"""
    raise ValueError("completion shell must be zsh or bash")
