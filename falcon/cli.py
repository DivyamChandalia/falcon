"""Command line interface for native Falcon."""

from __future__ import annotations

import argparse
import shlex
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

from . import __version__
from .commands import attach, clean, delete, logs, remember_job, top
from .completion import candidates, shell_script
from .config import DEFAULT_CONFIG, config_path, detect_shell, load_config, run_setup
from .dashboard import run_dashboard
from .launcher import build_jet_command, job_name, launch
from .resources import canonical_gpu, fetch_nodes, plan_resources


def resolve_preset(token: str, config: Dict[str, Any]) -> Optional[Tuple[str, int]]:
    """Resolve only configured preset names plus an arbitrary positive xN suffix."""
    lowered = token.lower()
    for name in config["presets"]:
        if lowered == name.lower():
            return name, 1
        prefix = name.lower() + "x"
        suffix = lowered[len(prefix):] if lowered.startswith(prefix) else ""
        if suffix.isdigit() and int(suffix) > 0:
            return name, int(suffix)
    return None


def _add_run_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("-c", "--cpu", help="Override CPU request:limit")
    parser.add_argument("-m", "--memory", help="Override memory request:limit")
    shm = parser.add_mutually_exclusive_group()
    shm.add_argument("--shm-size", help="Exact shared-memory size")
    shm.add_argument("--shm-percent", type=float, help="Shared memory as a percentage of allocated RAM")
    parser.add_argument("-j", "--job", help="Explicit Kubernetes job name")
    parser.add_argument(
        "--max", dest="maximize", action="store_true",
        help="Request 95%% of proportional node capacity instead of currently free CPU/RAM",
    )
    parser.add_argument("-a", "--async", dest="async_mode", action="store_true", help="Submit without following or cleanup")
    placement = parser.add_mutually_exclusive_group()
    placement.add_argument("--pin-node", action="store_true", help="Pin placement to Falcon's sizing node")
    placement.add_argument("--no-pin", dest="pin_node", action="store_false", help=argparse.SUPPRESS)
    parser.set_defaults(pin_node=False)
    parser.add_argument("--dry-run", action="store_true", help="Print generated Job YAML without submitting")
    parser.add_argument("--explain", action="store_true", help="Print the resolved Jet command")
    parser.add_argument("--jet-arg", action="append", default=[], help="Additional raw Jet argument (repeatable)")
    parser.add_argument("command", nargs=argparse.REMAINDER, help="Command after --; no command launches debug")


def _run_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="falcon PRESET", description="Run a cluster-aware GPU preset")
    _add_run_arguments(parser)
    return parser


def _legacy_run_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="falcon -j JOB -n N -g GPU", description="Legacy-compatible Falcon submission syntax"
    )
    parser.add_argument("-g", "--gpu-type", required=True, help="GPU model or configured preset name")
    parser.add_argument("-n", "--gpu", "--num-gpus", "--num_gpus", type=int, default=1, help="GPU count")
    _add_run_arguments(parser)
    return parser


def _print_plan(plan: Any, shm_size: str, pin_node: bool = False) -> None:
    if pin_node and plan.node:
        target = f"pinned to {plan.node}"
    elif plan.node:
        target = f"scheduler (sized from {plan.node})"
    elif plan.sizing_node:
        target = f"scheduler queue (sized from {plan.sizing_node})"
    else:
        target = "scheduler queue"
    state = "ready now" if plan.immediately_schedulable else "pending"
    print(
        f"[falcon] {plan.gpu_type} x{plan.gpu_count} | CPU {plan.cpu} | RAM {plan.memory} | "
        f"SHM {shm_size} | target {target} | {state}",
        flush=True,
    )
    if plan.warning:
        print(f"[falcon] WARNING: {plan.warning}", file=sys.stderr)


def _launch_request(preset_name: str, count: int, args: argparse.Namespace, config: Dict[str, Any]) -> int:
    preset = config["presets"][preset_name]
    try:
        nodes = fetch_nodes(config["cluster"]["kube_state_metrics_url"])
    except Exception as exc:
        raise ValueError(f"could not read cluster resources: {exc}") from exc
    plan = plan_resources(
        nodes=nodes,
        preset=preset_name,
        gpu_type=preset["gpu_type"],
        gpu_count=count,
        cpu_override=args.cpu,
        memory_override=args.memory,
        maximize=args.maximize,
    )
    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    name = args.job or job_name(command)
    jet_args: List[str] = []
    for item in args.jet_arg:
        jet_args.extend(shlex.split(item))
    invocation = build_jet_command(
        config=config,
        plan=plan,
        command=command,
        name=name,
        async_mode=args.async_mode,
        dry_run=args.dry_run,
        shm_size=args.shm_size,
        shm_percent=args.shm_percent,
        pin_node=args.pin_node,
        extra_jet_args=jet_args,
    )
    shm_index = invocation.index("--shm-size")
    _print_plan(plan, invocation[shm_index + 1], pin_node=args.pin_node)
    if args.explain:
        print(f"[falcon] {shlex.join(invocation)}", flush=True)
    remember_job(name)
    return launch(
        invocation,
        name,
        cleanup=bool(command) and not args.async_mode and not args.dry_run,
        namespace=config["cluster"]["namespace"],
    )


def run_preset(token: str, argv: Sequence[str], config: Dict[str, Any]) -> int:
    preset_name, count = resolve_preset(token, config) or (None, None)
    if preset_name is None:
        raise ValueError(f"unknown GPU preset: {token}")
    return _launch_request(preset_name, count, _run_parser().parse_args(list(argv)), config)


def run_legacy(argv: Sequence[str], config: Dict[str, Any]) -> int:
    args = _legacy_run_parser().parse_args(list(argv))
    requested = canonical_gpu(args.gpu_type)
    preset_name = next(
        (
            name for name, preset in config["presets"].items()
            if canonical_gpu(name) == requested or canonical_gpu(preset["gpu_type"]) == requested
        ),
        None,
    )
    if preset_name is None:
        choices = ", ".join(config["presets"])
        raise ValueError(f"unknown GPU type {args.gpu_type!r}; configured presets: {choices}")
    return _launch_request(preset_name, args.gpu, args, config)


def _looks_like_legacy_submission(argv: Sequence[str]) -> bool:
    before_command = list(argv[:argv.index("--")]) if "--" in argv else list(argv)
    flags = {
        "-j", "--job", "-g", "--gpu-type", "-n", "--gpu", "--num-gpus", "--num_gpus",
        "-c", "--cpu", "-m", "--memory", "--shm-size", "--shm-percent", "-a", "--async",
        "--max",
    }
    if not before_command:
        return False
    first = before_command[0]
    return first in flags or any(first.startswith(flag + "=") for flag in flags if flag.startswith("--"))


def _main_parser(config: Dict[str, Any]) -> argparse.ArgumentParser:
    preset_help = ", ".join(f"{name}[xN]" for name in config["presets"])
    parser = argparse.ArgumentParser(
        prog="falcon",
        description="Native cluster-aware ML jobs",
        epilog=f"GPU presets: {preset_help}. Example: falcon 2080tix3 -- python train.py",
    )
    parser.add_argument("--version", action="version", version=f"falcon {__version__}")
    parser.add_argument("--config", help="Config path (default: ~/.falconrc or FALCON_CONFIG)")
    sub = parser.add_subparsers(dest="command")
    setup = sub.add_parser("setup", help="Write .falconrc and install shell completion")
    setup.add_argument("--force", action="store_true")
    setup.add_argument("--non-interactive", action="store_true")
    setup.add_argument("--no-shell", action="store_true", help="Do not update the active shell rc file")
    dashboard = sub.add_parser("dashboard", aliases=["dash"], help="Open the nvitop-style dashboard")
    dashboard.add_argument("--once", action="store_true", help="Print one compact snapshot and exit")
    dashboard.add_argument("--json", action="store_true", help="Print one machine-readable JSON snapshot")
    dashboard.add_argument("--job", help="Show metrics only for this job")
    dashboard.add_argument("--samples", type=int, help="Snapshot sample count (agent default: 5)")
    dashboard.add_argument("--interval", type=float, default=1.0, help="Seconds between snapshot samples")
    for name, help_text in (
        ("logs", "Follow job logs"), ("attach", "Attach to a job"), ("top", "Run nvitop in a job"),
    ):
        command = sub.add_parser(name, help=help_text)
        command.add_argument("job", nargs="?", help="Job name (defaults to last Falcon job)")
    remove = sub.add_parser("delete", aliases=["kill"], help="Delete one or more jobs")
    remove.add_argument("jobs", nargs="*", help="Job names (defaults to last Falcon job)")
    cleaner = sub.add_parser("clean", help="Delete completed and failed jobs")
    init = sub.add_parser("shell-init", help="Print native wrapper and completion")
    init.add_argument("shell", nargs="?", choices=["zsh", "bash"])
    completion = sub.add_parser("completion", help="Print completion for a shell")
    completion.add_argument("shell", choices=["zsh", "bash"])
    sub.add_parser("config", help="Print the active config path")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    config_arg = None
    if len(argv) >= 2 and argv[0] == "--config":
        config_arg, argv = argv[1], argv[2:]
    elif argv and argv[0].startswith("--config="):
        config_arg, argv = argv[0].split("=", 1)[1], argv[1:]
    try:
        bootstrap_command = argv[0] if argv else ""
        try:
            config = load_config(config_arg)
        except ValueError:
            # Setup must be able to replace an obsolete config, and shell init
            # must remain available while that repair is pending.
            if bootstrap_command not in {"setup", "shell-init", "completion", "config"}:
                raise
            config = DEFAULT_CONFIG
        if _looks_like_legacy_submission(argv):
            return run_legacy(argv, config)
        if argv and resolve_preset(argv[0], config):
            return run_preset(argv[0], argv[1:], config)
        if argv and argv[0] == "_complete":
            kind = argv[1] if len(argv) > 1 else "commands"
            if kind not in {"commands", "jobs", "options"}:
                return 2
            subject = argv[2] if len(argv) > 2 else ""
            print("\n".join(candidates(kind, config, subject)))
            return 0
        parser = _main_parser(config)
        args = parser.parse_args((["--config", config_arg] if config_arg else []) + argv)
        active_path = args.config or config_arg
        namespace = config["cluster"]["namespace"]
        if args.command == "setup":
            target, rc_path = run_setup(
                active_path, force=args.force, non_interactive=args.non_interactive, install_shell=not args.no_shell
            )
            print(f"Wrote {target}")
            if rc_path:
                print(f"Installed Falcon and completion in {rc_path}; open a new shell or source it.")
            return 0
        if args.command in {"dashboard", "dash"}:
            run_dashboard(
                load_config(active_path), namespace, once=args.once, json_output=args.json,
                job=args.job, samples=args.samples, sample_interval=args.interval,
            )
            return 0
        if args.command == "logs":
            return logs(namespace, args.job)
        if args.command == "attach":
            return attach(namespace, args.job)
        if args.command == "top":
            return top(namespace, args.job)
        if args.command in {"delete", "kill"}:
            return delete(namespace, args.jobs)
        if args.command == "clean":
            return clean(namespace)
        if args.command in {"shell-init", "completion"}:
            shell = args.shell or detect_shell()[0]
            print(shell_script(shell, config=config))
            return 0
        if args.command == "config":
            print(config_path(active_path))
            return 0
        parser.print_help()
        return 0
    except KeyboardInterrupt:
        return 130
    except (FileExistsError, FileNotFoundError, ValueError, OSError) as exc:
        print(f"falcon: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
