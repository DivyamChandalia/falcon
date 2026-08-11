"""Falcon's human-friendly and machine-readable command line interface."""

from __future__ import annotations

import argparse
import getpass
import shlex
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import yaml

from . import __version__
from .agent_skills import (
    detect_agents,
    install_skills,
    uninstall_skills,
)
from .cluster import (
    ClusterCollector,
    ClusterSnapshot,
    JobSnapshot,
    build_cluster_snapshot,
    build_job_snapshot,
    build_job_snapshots,
)
from .commands import attach, clean, kill, remember_job, target_job, top
from .completion import COMMAND_ALIASES, shell_script
from .config import (
    config_path,
    detect_shell,
    load_config,
    logname,
    run_setup,
    save_resources_view,
)
from .coder import (
    CoderAuthenticationRequired,
    CoderClient,
    CoderError,
    build_access_links,
    build_parameter_values,
    generate_workspace_name,
    parse_parameter_overrides,
    resolve_connection,
    resolve_template,
    select_access_links,
    save_connection,
    validate_workspace_name,
    workspace_job_name,
)
from .dashboard import UsageCollector, run_dashboard
from .demo import DemoCollector
from .kubernetes import KubernetesClient, KubernetesError
from .launcher import (
    build_specification,
    job_name,
    resolve_environment,
    submit,
)
from .models import GPURequest, JobRequest, NodeResources
from .output import dumps, render_table
from .planning import canonical_gpu, plan_cpu_resources, plan_resources
from .resources import MetricsClusterCollector, fetch_nodes
from .resources_ui import FalconResourcesApp
from .theme import COLOR_MODES

EXIT_USAGE = 2
EXIT_KUBERNETES = 3
EXIT_NOT_FOUND = 4
EXIT_CONFLICT = 5
EXIT_CODER = 6

_LAUNCH_SENTINEL = "__falcon_launch__"


class CliError(ValueError):
    def __init__(self, message: str, code: int = EXIT_USAGE) -> None:
        super().__init__(message)
        self.code = code


def resolve_preset(
    token: str, config: Mapping[str, Any]
) -> Optional[Tuple[str, int]]:
    lowered = token.lower()
    for name in config.get("presets", {}):
        normalized = name.lower()
        if lowered == normalized:
            return name, 1
        prefix = normalized + "x"
        suffix = lowered[len(prefix) :] if lowered.startswith(prefix) else ""
        if suffix.isdigit() and int(suffix) > 0:
            return name, int(suffix)
    return None


def _output(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--output",
        choices=("human", "json"),
        default="human",
        help="Output format (default: human)",
    )


def _color(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--color",
        choices=COLOR_MODES,
        default=None,
        metavar="MODE",
        help=(
            "TUI colour mode: truecolor (default), 256, 16, or auto; "
            "FALCON_COLOR may set the default"
        ),
    )


def _namespace(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--namespace",
        help="Override the configured namespace for this command",
    )


def _submission_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--gpu", help="GPU model or configured preset")
    parser.add_argument("--gpus", "-n", type=int, default=1, help="GPU count")
    parser.add_argument("--cpu", "-c", help="CPU request, e.g. 8 or 8:8")
    parser.add_argument("--memory", "-m", help="RAM request, e.g. 32Gi")
    parser.add_argument("--name", "-j", help="Kubernetes Job name")
    parser.add_argument(
        "--environment",
        help="Conda/venv path, 'auto' (default), or 'none'",
    )
    parser.add_argument("--image", help="Override the configured container image")
    parser.add_argument(
        "--env",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Container environment variable (repeatable)",
    )
    shared = parser.add_mutually_exclusive_group()
    shared.add_argument("--shm-size", help="Exact /dev/shm size")
    shared.add_argument(
        "--shm-percent",
        type=float,
        help="Shared memory as a percentage of requested RAM",
    )
    parser.add_argument(
        "--max",
        dest="maximize",
        action="store_true",
        help="Size from 95%% of proportional node capacity",
    )
    parser.add_argument(
        "--pin-node",
        action="store_true",
        help="Pin to the node used for resource planning",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the structured Kubernetes manifest without creating a Job",
    )
    parser.add_argument(
        "--async",
        dest="legacy_async",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "-f",
        "--follow",
        action="store_true",
        help="Stay attached to Job logs; Ctrl+C kills this Job",
    )
    _namespace(parser)
    _output(parser)
    parser.add_argument(
        "command",
        nargs=argparse.REMAINDER,
        help="Command argv after --; omit for a debug container",
    )


def _parser(config: Mapping[str, Any]) -> argparse.ArgumentParser:
    presets = ", ".join(config.get("presets", {}))
    parser = argparse.ArgumentParser(
        prog="falcon",
        description="Run Kubernetes Jobs like local commands",
        epilog=(
            f"GPU presets: {presets}. Example: "
            "falcon h100x2 -- python train.py"
        ),
    )
    parser.add_argument(
        "--version", action="version", version=f"falcon {__version__}"
    )
    parser.add_argument(
        "--config",
        help="Config path (default: ~/.falconrc or FALCON_CONFIG)",
    )
    sub = parser.add_subparsers(dest="command_name")

    jobs = sub.add_parser("jobs", help="List a bounded set of Jobs")
    jobs.add_argument("--limit", type=int, default=50)
    jobs.add_argument("--status")
    jobs.add_argument("--gpu")
    jobs.add_argument("--node")
    _namespace(jobs)
    _output(jobs)

    get = sub.add_parser("get", help="Inspect one Job and its attempts")
    get.add_argument("job")
    _namespace(get)
    _output(get)

    event_parser = sub.add_parser("events", help="Show bounded Job events")
    event_parser.add_argument("job")
    event_parser.add_argument("--limit", type=int, default=50)
    event_parser.add_argument(
        "-f",
        "--follow",
        action="store_true",
        help="Keep polling and print new or updated events",
    )
    _namespace(event_parser)
    _output(event_parser)

    log_parser = sub.add_parser("logs", help="Read bounded Job logs")
    log_parser.add_argument("job", nargs="?")
    log_parser.add_argument("--tail", type=int, default=100)
    log_follow = log_parser.add_mutually_exclusive_group()
    log_follow.add_argument(
        "--follow",
        dest="follow",
        action="store_true",
        help="Keep streaming logs (default for human output)",
    )
    log_follow.add_argument(
        "--no-follow",
        dest="follow",
        action="store_false",
        help="Print the requested tail and exit",
    )
    log_parser.set_defaults(follow=None)
    log_parser.add_argument("--container")
    _namespace(log_parser)
    _output(log_parser)

    attach_parser = sub.add_parser("attach", help="Attach to a running Job Pod")
    attach_parser.add_argument("job", nargs="?")
    _namespace(attach_parser)

    top_parser = sub.add_parser(
        "top",
        help="Open nvitop for a running Job",
    )
    top_parser.add_argument("job", nargs="?")
    _namespace(top_parser)

    metrics = sub.add_parser(
        "metrics",
        help="Return JSON GPU, VRAM, CPU, and memory utilization",
    )
    metrics.add_argument("job")
    metrics.add_argument(
        "--interval",
        type=float,
        default=10.0,
        help="Observation duration in seconds (1-300; default: 10)",
    )
    _namespace(metrics)
    metrics.add_argument(
        "--output",
        choices=("json",),
        default="json",
        help=argparse.SUPPRESS,
    )

    killer = sub.add_parser(
        "kill", help="Kill Jobs or remove Coder-owned workspaces"
    )
    killer.add_argument("jobs", nargs="*")
    _namespace(killer)
    _output(killer)

    cleaner = sub.add_parser("clean", help="Delete succeeded Jobs")
    _namespace(cleaner)

    dashboard = sub.add_parser("dashboard", help="Open the Job dashboard")
    dashboard.add_argument("--job")
    dashboard.add_argument(
        "--demo",
        nargs="?",
        const="mixed",
        help="Use deterministic demo data (optional state)",
    )
    _color(dashboard)

    resources = sub.add_parser(
        "resources", help="Inspect cluster request headroom and node consumers"
    )
    resources.add_argument("--node")
    resources.add_argument("--gpu")
    resources.add_argument("--limit", type=int, default=100)
    resources.add_argument(
        "--consumer-limit",
        type=int,
        default=100,
        help="Maximum workloads included per node in JSON output (default: 100)",
    )
    resources.add_argument(
        "--demo",
        nargs="?",
        const="mixed",
        help="Use deterministic demo data (optional state)",
    )
    _color(resources)
    _namespace(resources)
    _output(resources)

    coder = sub.add_parser(
        "coder", help="Create a sized Coder workspace and print access links"
    )
    coder.add_argument(
        "preset",
        nargs="?",
        metavar="PRESET_OR_WORKSPACE",
        help=(
            "GPU preset to create, or an existing workspace/Job name whose "
            "access links should be printed"
        ),
    )
    coder.add_argument(
        "--cpu", "-c",
        help="CPU request[:limit], or a GPU-preset sizing override",
    )
    coder.add_argument(
        "--memory", "-m",
        help="RAM request[:limit], or a GPU-preset sizing override",
    )
    coder.add_argument(
        "--name", "-j",
        help="Workspace name (default: random color-animal-number)",
    )
    coder.add_argument(
        "--template",
        help="Coder template name or ID (default: coder.template, IDEs)",
    )
    coder.add_argument(
        "--url", help="Coder deployment URL (overrides CODER_URL and config)"
    )
    coder.add_argument(
        "--parameter",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="Additional Coder template parameter (repeatable)",
    )
    coder.add_argument(
        "--access",
        default="all",
        metavar="APP",
        help=(
            "Links to print: all (default), terminal, vscode, cursor, "
            "jupyter, antigravity, Antigravity 2.0, Antigravity 2.0 IDE, "
            "or an app slug"
        ),
    )
    coder.add_argument(
        "--timeout",
        type=float,
        help="Seconds to wait for the Coder agent (default: config, 600)",
    )

    setup = sub.add_parser(
        "setup", help="Create config, completion, and optional agent skills"
    )
    setup.add_argument("--force", action="store_true")
    setup.add_argument("--non-interactive", action="store_true")
    setup.add_argument("--no-shell", action="store_true")
    skills = setup.add_mutually_exclusive_group()
    skills.add_argument(
        "--install-skills",
        metavar="AGENTS",
        help="Comma-separated: codex,claude,opencode",
    )
    skills.add_argument("--skip-skills", action="store_true")
    skills.add_argument(
        "--uninstall-skills",
        metavar="AGENTS",
        help="Remove unchanged Falcon-owned skill copies",
    )

    completion = sub.add_parser("completion", help="Print shell completion")
    completion.add_argument("shell", nargs="?", choices=("bash", "zsh"))
    sub.add_parser("config", help="Print the active config path")
    return parser


def _launch_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="falcon",
        description=(
            "Launch a Kubernetes Job directly with a GPU preset or explicit "
            "CPU and memory"
        ),
    )
    parser.add_argument("--config", help=argparse.SUPPRESS)
    _submission_arguments(parser)
    return parser


def _config_argument(argv: Sequence[str]) -> Optional[str]:
    for index, token in enumerate(argv):
        if token == "--config" and index + 1 < len(argv):
            return argv[index + 1]
        if token.startswith("--config="):
            return token.split("=", 1)[1]
    return None


def _rewrite_shorthand(
    argv: List[str], config: Mapping[str, Any]
) -> List[str]:
    prefix: List[str] = []
    remaining = list(argv)
    if len(remaining) >= 2 and remaining[0] == "--config":
        prefix, remaining = remaining[:2], remaining[2:]
    elif remaining and remaining[0].startswith("--config="):
        prefix, remaining = remaining[:1], remaining[1:]
    if not remaining:
        return argv
    public_commands = {
        "jobs", "get", "events", "logs", "attach", "top", "metrics",
        "kill", "clean", "dashboard", "resources", "coder", "setup",
        "completion", "config", "shell-init",
    }
    # Falcon 0.1 installed this command in shell startup files. Keep a hidden,
    # output-compatible migration alias so upgrading the executable cannot
    # disable Tab completion before the user next runs ``falcon setup``.
    if remaining[0] == "shell-init":
        return [*prefix, "completion", *remaining[1:]]
    if remaining[0] == "submit":
        raise CliError(
            "'falcon submit' was removed; use 'falcon h100[xN] -- COMMAND' "
            "for GPU Jobs or 'falcon -c CPU -m MEMORY -- COMMAND' for CPU Jobs"
        )
    if remaining[0] == "delete":
        raise CliError("'falcon delete' was removed; use 'falcon kill JOB'")
    if remaining[0] in COMMAND_ALIASES:
        return [
            *prefix,
            COMMAND_ALIASES[remaining[0]],
            *remaining[1:],
        ]
    if remaining[0] in public_commands or remaining[0] in {"-h", "--help", "--version"}:
        return argv
    resolved = resolve_preset(remaining[0], config)
    if resolved:
        preset, count = resolved
        return [
            *prefix,
            _LAUNCH_SENTINEL,
            "--gpu",
            preset,
            "--gpus",
            str(count),
            *remaining[1:],
        ]
    before_command = (
        remaining[: remaining.index("--")] if "--" in remaining else remaining
    )
    if any(
        token in {
            "-c", "--cpu", "-m", "--memory", "--gpu", "--gpus",
            "-n",
        }
        for token in before_command
    ):
        return [*prefix, _LAUNCH_SENTINEL, *remaining]
    if any(token in {"-g", "--gpu-type"} for token in before_command):
        rewritten = [_LAUNCH_SENTINEL]
        iterator = iter(remaining)
        for token in iterator:
            if token in {"-g", "--gpu-type"}:
                rewritten += ["--gpu", next(iterator, "")]
            elif token in {"-n", "--num-gpus", "--num_gpus"}:
                rewritten += ["--gpus", next(iterator, "")]
            else:
                rewritten.append(token)
        return [*prefix, *rewritten]
    return argv


def _namespace_value(args: argparse.Namespace, config: Mapping[str, Any]) -> str:
    return args.namespace or str(config["cluster"]["namespace"])


def _parse_env(values: Iterable[str]) -> Dict[str, str]:
    result: Dict[str, str] = {}
    for value in values:
        key, separator, setting = value.partition("=")
        if not separator or not key:
            raise CliError(f"invalid --env {value!r}; expected KEY=VALUE")
        result[key] = setting
    return result


def _planning_nodes(
    config: Mapping[str, Any],
    client: KubernetesClient,
) -> List[NodeResources]:
    metrics_url = config.get("cluster", {}).get("kube_state_metrics_url")
    if metrics_url:
        try:
            return fetch_nodes(str(metrics_url), timeout=8)
        except Exception:
            # The Kubernetes inventory path retains honest request semantics
            # and gives a clearer RBAC/API error if both sources are absent.
            pass
    snapshot = build_cluster_snapshot(
        nodes=client.list_nodes(),
        pods=client.list_pods(""),
    )
    return [
        NodeResources(
            name=node.name,
            cpu_total=node.allocatable.cpu_cores,
            cpu_used=node.requested.cpu_cores,
            memory_total_gib=node.allocatable.memory_gib,
            memory_used_gib=node.requested.memory_gib,
            gpu_total=node.allocatable.gpu_count,
            gpu_used=node.requested.gpu_count,
            gpu_product=node.gpu_model or "",
            unschedulable=not node.schedulable or node.ready is False,
        )
        for node in snapshot.nodes
    ]


def _print_submission_summary(
    request: JobRequest,
    plan,
    specification,
    *,
    stream,
) -> None:
    """Show the resolved request immediately before the Kubernetes create."""

    container = (
        specification.manifest.get("spec", {})
        .get("template", {})
        .get("spec", {})
        .get("containers", [{}])[0]
    )
    resources = container.get("resources", {}).get("requests", {})
    gpu = next(
        (
            f"{key}={value}"
            for key, value in resources.items()
            if key.endswith("/gpu")
        ),
        "gpu=-",
    )
    command = shlex.join(request.command) if request.command else "<interactive shell>"
    image = container.get("image") or request.image or "-"
    extras = []
    if plan.compute.shared_memory:
        extras.append(f"shm={plan.compute.shared_memory}")
    if request.environment is not None:
        extras.append(f"environment={request.environment.path}")
    suffix = f" · {' · '.join(extras)}" if extras else ""
    print(
        f"Submitting Job request · {request.name} · namespace={request.namespace} · "
        f"image={image} · {gpu} · cpu={resources.get('cpu', plan.compute.cpu)} · "
        f"memory={resources.get('memory', plan.compute.memory)}{suffix} · "
        f"command={command}",
        file=stream,
    )


def _submit_command(
    args: argparse.Namespace,
    config: Mapping[str, Any],
) -> int:
    if args.follow and args.output == "json":
        raise CliError("--follow cannot be combined with --output json")
    namespace = _namespace_value(args, config)
    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    interactive_debug = not command
    if interactive_debug and args.output == "json" and not args.dry_run:
        raise CliError(
            "interactive debug sessions require human output; "
            "use --dry-run --output json to inspect the manifest"
        )
    name = args.name or job_name(command)
    gpu = None
    if args.gpu:
        preset = config.get("presets", {}).get(args.gpu)
        model = preset.get("gpu_type") if isinstance(preset, Mapping) else args.gpu
        gpu = GPURequest(canonical_gpu(str(model)), args.gpus)
    elif args.gpus != 1:
        raise CliError("--gpus requires --gpu")

    percent = (
        args.shm_percent
        if args.shm_percent is not None
        else config.get("resources", {}).get("shared_memory_percent", 15)
    )
    if gpu is None:
        if not args.cpu or not args.memory:
            raise CliError("CPU-only Jobs require both --cpu and --memory")
        plan = plan_cpu_resources(
            args.cpu,
            args.memory,
            shared_memory=args.shm_size,
            shared_memory_percent=percent,
        )
    else:
        client = KubernetesClient(namespace)
        nodes = _planning_nodes(config, client)
        plan = plan_resources(
            nodes,
            args.gpu,
            gpu.model,
            gpu.count,
            cpu_override=args.cpu,
            memory_override=args.memory,
            maximize=args.maximize,
            shared_memory=args.shm_size,
            shared_memory_percent=percent,
        )
    environment = resolve_environment(args.environment, config)
    request = JobRequest(
        name=name,
        namespace=namespace,
        command=tuple(command),
        gpu=gpu,
        compute=plan.compute,
        environment=environment,
        image=args.image,
        working_dir=str(Path.cwd()),
        env=_parse_env(args.env),
        annotations={"falcon.dev/owner": logname()},
        pin_node=args.pin_node,
    )
    specification = build_specification(request, plan, config)
    if args.dry_run:
        if args.output == "json":
            print(
                dumps(
                    "JobDryRun",
                    {
                        "request": request,
                        "resource_plan": plan,
                        "manifest": specification.manifest,
                    },
                )
            )
        else:
            print(yaml.safe_dump(specification.manifest, sort_keys=False), end="")
        return 0
    _print_submission_summary(
        request,
        plan,
        specification,
        stream=sys.stderr if args.output == "json" else sys.stdout,
    )
    client = KubernetesClient(namespace)
    result = submit(specification, client)
    remember_job(result.name)
    if args.output == "json":
        print(dumps("SubmittedJob", result))
    elif not interactive_debug:
        print(f"Submitted Job {result.name} in namespace {result.namespace}.")
        if plan.warning:
            print(f"Warning: {plan.warning}", file=sys.stderr)
    if interactive_debug:
        shell, shell_rc = detect_shell()
        prompt_label = plan.preset
        if plan.gpu is not None and plan.gpu.count > 1:
            prompt_label = f"{prompt_label}x{plan.gpu.count}"
        gpu_request = (
            f"{plan.gpu.model} x{plan.gpu.count}"
            if plan.gpu is not None
            else "-"
        )
        requested = [
            f"GPU {gpu_request}",
            f"CPU {plan.compute.cpu}",
            f"RAM {plan.compute.memory}",
        ]
        if plan.compute.shared_memory:
            requested.append(f"SHM {plan.compute.shared_memory}")
        print(f"Waiting for debug Pod · {' · '.join(requested)}...")
        if plan.warning:
            print(f"Warning: {plan.warning}", file=sys.stderr)
        session_error: Optional[BaseException] = None
        cleanup_error: Optional[KubernetesError] = None
        try:
            connected = client.exec_shell(
                result.name,
                shell,
                prompt_label=prompt_label,
                rc_path=str(shell_rc),
            )
            if connected.returncode:
                raise KubernetesError(
                    connected.stderr.strip()
                    or connected.stdout.strip()
                    or f"interactive shell exited {connected.returncode}",
                    returncode=connected.returncode,
                    stderr=connected.stderr,
                    command=connected.argv,
                )
        except (KeyboardInterrupt, KubernetesError) as exc:
            session_error = exc
        finally:
            print(f"Deleting debug Job {result.name}...")
            try:
                client.delete_jobs([result.name], wait=False)
            except KubernetesError as exc:
                cleanup_error = exc
        if cleanup_error is not None:
            if session_error is not None:
                print(
                    f"falcon: could not delete debug Job {result.name}: "
                    f"{cleanup_error}",
                    file=sys.stderr,
                )
            else:
                raise cleanup_error
        if session_error is not None:
            raise session_error
    elif args.follow:
        try:
            streamed = client.logs(result.name, tail=100, follow=True)
        except KeyboardInterrupt:
            print(
                f"\nStopping and killing Job {result.name}...",
                file=sys.stderr,
            )
            try:
                client.delete_jobs([result.name], wait=False)
            except KubernetesError as exc:
                print(
                    f"falcon: could not kill Job {result.name}: {exc}",
                    file=sys.stderr,
                )
            raise
        if streamed.returncode:
            raise KubernetesError(
                streamed.stderr.strip()
                or streamed.stdout.strip()
                or f"logs exited {streamed.returncode}",
                returncode=streamed.returncode,
                stderr=streamed.stderr,
                command=streamed.argv,
            )
    return 0


def _job_inventory(client: KubernetesClient) -> Tuple[JobSnapshot, ...]:
    return build_job_snapshots(
        {
            "jobs": client.list_jobs(),
            "pods": client.list_pods(),
        }
    )


def _gpu_text(job: JobSnapshot, allocated: bool = False) -> str:
    vector = job.allocated if allocated else job.requested
    return (
        f"{vector.gpu_model or 'unknown'} x{vector.gpu_count}"
        if vector.gpu_count
        else "-"
    )


def _memory_gib(value: int) -> str:
    return f"{value / (1024**3):.1f}Gi"


def _jobs_command(
    args: argparse.Namespace,
    config: Mapping[str, Any],
) -> int:
    if not 1 <= args.limit <= 500:
        raise CliError("--limit must be between 1 and 500")
    client = KubernetesClient(_namespace_value(args, config))
    jobs = list(_job_inventory(client))
    if args.status:
        jobs = [job for job in jobs if job.status.lower() == args.status.lower()]
    if args.gpu:
        wanted = canonical_gpu(args.gpu)
        jobs = [
            job for job in jobs
            if job.requested.gpu_count
            and canonical_gpu(job.requested.gpu_model or "") == wanted
        ]
    if args.node:
        jobs = [job for job in jobs if args.node in job.nodes]
    jobs.sort(key=lambda job: job.created_at or "", reverse=True)
    jobs = jobs[: args.limit]
    if args.output == "json":
        print(
            dumps(
                "JobList",
                jobs,
                meta={"count": len(jobs), "limit": args.limit},
            )
        )
        return 0
    print(
        render_table(
            ("NAME", "STATUS", "GPU REQUESTED", "GPU ALLOCATED", "CPU", "RAM", "ATTEMPTS"),
            (
                (
                    job.name,
                    job.status,
                    _gpu_text(job),
                    _gpu_text(job, allocated=True),
                    f"{job.requested.cpu_cores:g}",
                    _memory_gib(job.requested.memory_bytes),
                    f"{job.attempts.pod_attempts} ({job.attempts.failed_attempts} failed)",
                )
                for job in jobs
            ),
            maximum_widths=(40, 12, 20, 20, 8, 10, 18),
        )
    )
    return 0


def _event_data(values: Iterable[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    result = []
    for item in values:
        metadata = item.get("metadata") or {}
        series = item.get("series") or {}
        involved = item.get("involvedObject") or {}
        result.append(
            {
                "timestamp": (
                    item.get("eventTime")
                    or item.get("lastTimestamp")
                    or series.get("lastObservedTime")
                    or metadata.get("creationTimestamp")
                ),
                "type": item.get("type") or "Normal",
                "reason": item.get("reason") or "Unknown",
                "message": item.get("message") or "",
                "object": involved.get("name"),
                "count": item.get("count") or series.get("count") or 1,
            }
        )
    return result


def _one_job(
    client: KubernetesClient,
    name: str,
) -> Tuple[JobSnapshot, List[Dict[str, Any]]]:
    job = client.get_json("job.batch", name)
    pods = client.job_pods(name)
    return build_job_snapshot(job, pods), pods


def _get_command(
    args: argparse.Namespace,
    config: Mapping[str, Any],
) -> int:
    client = KubernetesClient(_namespace_value(args, config))
    job, _ = _one_job(client, args.job)
    data = {"job": job}
    if args.output == "json":
        print(dumps("Job", data))
        return 0
    attempt = job.attempts
    rows = [
        ("Job", job.name),
        ("Status", job.status),
        ("GPU requested", _gpu_text(job)),
        ("GPU allocated", _gpu_text(job, allocated=True)),
        ("Node", ", ".join(job.nodes) or "-"),
        ("CPU requested", f"{job.requested.cpu_cores:g}"),
        ("RAM requested", _memory_gib(job.requested.memory_bytes)),
        ("Container restarts", str(attempt.container_restarts)),
        ("Pod attempts", str(attempt.pod_attempts)),
        ("Succeeded attempts", str(attempt.succeeded_attempts)),
        ("Failed attempts", str(attempt.failed_attempts)),
        ("Active Pod", attempt.active_pod or "-"),
        ("Backoff limit", "-" if attempt.backoff_limit is None else str(attempt.backoff_limit)),
        ("Command", " ".join(job.command) or "-"),
    ]
    print(render_table(("FIELD", "VALUE"), rows, maximum_widths=(22, 100)))
    return 0


def _event_key(event: Mapping[str, Any]) -> Tuple[Any, ...]:
    return (
        event.get("timestamp"),
        event.get("type"),
        event.get("reason"),
        event.get("object"),
        event.get("message"),
        event.get("count"),
    )


def _render_event_table(events: Iterable[Mapping[str, Any]]) -> str:
    return render_table(
        ("TIME", "TYPE", "REASON", "OBJECT", "MESSAGE"),
        (
            (
                event["timestamp"] or "-",
                event["type"],
                event["reason"],
                event["object"] or "-",
                event["message"],
            )
            for event in events
        ),
        maximum_widths=(24, 9, 22, 30, 80),
    )


def _events_command(
    args: argparse.Namespace,
    config: Mapping[str, Any],
) -> int:
    if not 1 <= args.limit <= 200:
        raise CliError("--limit must be between 1 and 200")
    if args.follow and args.output == "json":
        raise CliError("--follow cannot be combined with --output json")
    client = KubernetesClient(_namespace_value(args, config))
    job, pods = _one_job(client, args.job)
    names = [job.name] + [
        str((pod.get("metadata") or {}).get("name") or "") for pod in pods
    ]
    events = _event_data(client.events(names, limit=args.limit))
    if args.output == "json":
        print(
            dumps(
                "EventList",
                events,
                meta={"job": job.name, "count": len(events), "limit": args.limit},
            )
        )
    else:
        print(_render_event_table(events))
        if args.follow:
            seen = {_event_key(event) for event in events}
            while True:
                time.sleep(1.0)
                job, pods = _one_job(client, args.job)
                names = [job.name] + [
                    str((pod.get("metadata") or {}).get("name") or "")
                    for pod in pods
                ]
                current = _event_data(
                    client.events(names, limit=args.limit)
                )
                additions = [
                    event
                    for event in current
                    if _event_key(event) not in seen
                ]
                if additions:
                    print(_render_event_table(additions))
                seen.update(_event_key(event) for event in current)
    return 0


def _logs_command(
    args: argparse.Namespace,
    config: Mapping[str, Any],
) -> int:
    if not 0 <= args.tail <= 100_000:
        raise CliError("--tail must be between 0 and 100000")
    if args.follow is True and args.output == "json":
        raise CliError("--follow cannot be combined with --output json")
    follow = (
        args.follow
        if args.follow is not None
        else args.output == "human"
    )
    name = target_job(args.job)
    client = KubernetesClient(_namespace_value(args, config))
    result = client.logs(
        name,
        tail=args.tail,
        follow=follow,
        container=args.container,
    )
    if result.returncode:
        raise KubernetesError(
            result.stderr.strip() or result.stdout.strip()
            or f"logs exited {result.returncode}",
            returncode=result.returncode,
            stderr=result.stderr,
            command=result.argv,
        )
    remember_job(name)
    if args.output == "json":
        lines = result.stdout.splitlines()
        print(
            dumps(
                "JobLogs",
                {"job": name, "lines": lines},
                meta={"tail": args.tail, "count": len(lines)},
            )
        )
    elif not follow and result.stdout:
        print(result.stdout, end="" if result.stdout.endswith("\n") else "\n")
    return 0


def _percent_summary(values: Sequence[float]) -> Dict[str, Any]:
    if not values:
        return {
            "current": None,
            "average": None,
            "minimum": None,
            "maximum": None,
            "samples": 0,
        }
    return {
        "current": values[-1],
        "average": sum(values) / len(values),
        "minimum": min(values),
        "maximum": max(values),
        "samples": len(values),
    }


def _observation_offsets(duration: float) -> List[float]:
    offsets = [0.0]
    second = 1.0
    while second < duration:
        offsets.append(second)
        second += 1.0
    offsets.append(duration)
    return offsets


def _metrics_command(
    args: argparse.Namespace,
    config: Mapping[str, Any],
) -> int:
    if not 1 <= args.interval <= 300:
        raise CliError("--interval must be between 1 and 300 seconds")

    requested_job = args.job
    namespace = _namespace_value(args, config)
    thresholds = {
        canonical_gpu(str(preset["gpu_type"])): float(
            preset.get("minimum_utilization", 30)
        )
        for preset in config.get("presets", {}).values()
    }
    dashboard = config.get("dashboard", {})
    collector = UsageCollector(
        namespace,
        thresholds,
        float(dashboard.get("ema_alpha", 0.1)),
        job_filter=requested_job,
        streaming_gpu=True,
        metrics_url=None,
        collect_availability=False,
    )
    samples: Dict[str, List[float]] = {
        "gpu": [],
        "vram": [],
        "cpu": [],
        "memory": [],
    }
    latest = None
    offsets = _observation_offsets(args.interval)
    try:
        previous_offset = 0.0
        for offset in offsets:
            if offset:
                time.sleep(offset - previous_offset)
            previous_offset = offset
            rows = collector.collect()
            row = next((item for item in rows if item.job == requested_job), None)
            if row is None:
                continue
            latest = row
            if row.gpu_allocated_count > 0 and row.gpu_util is not None:
                samples["gpu"].append(float(row.gpu_util))
            if (
                row.gpu_allocated_count > 0
                and row.gpu_metrics_available
                and row.gpu_memory_total_gib > 0
            ):
                samples["vram"].append(
                    row.gpu_memory_used_gib / row.gpu_memory_total_gib * 100
                )
            if row.cpu_metrics_available and row.cpu_allocated > 0:
                samples["cpu"].append(row.cpu_used / row.cpu_allocated * 100)
            if row.cpu_metrics_available and row.memory_allocated_gib > 0:
                samples["memory"].append(
                    row.memory_used_gib / row.memory_allocated_gib * 100
                )
    finally:
        collector.close()

    if latest is None:
        detail = collector.last_error or f"Job {requested_job!r} was not found"
        raise KubernetesError(detail)
    gpu_model = (
        latest.gpu_allocated_type
        if latest.gpu_allocated_count
        else None
    )
    floor = (
        thresholds.get(canonical_gpu(gpu_model), 30.0)
        if gpu_model
        else None
    )
    utilization = {
        "gpu_percent": _percent_summary(samples["gpu"]),
        "vram_percent": _percent_summary(samples["vram"]),
        "cpu_percent": _percent_summary(samples["cpu"]),
        "memory_percent": _percent_summary(samples["memory"]),
    }
    gpu_average = utilization["gpu_percent"]["average"]
    data = {
        "job": requested_job,
        "status": latest.status,
        "allocation": {
            "gpu": {
                "model": gpu_model,
                "count": int(latest.gpu_allocated_count),
            },
            "cpu_cores": float(latest.cpu_allocated),
            "memory_bytes": int(latest.memory_allocated_gib * 1024**3),
            "vram_bytes": (
                int(latest.gpu_memory_total_gib * 1024**3)
                if latest.gpu_metrics_available
                and latest.gpu_memory_total_gib > 0
                else None
            ),
        },
        "utilization": utilization,
        "eviction_policy": {
            "minimum_average_gpu_utilization_percent": floor,
            "observed_average_meets_minimum": (
                None
                if floor is None or gpu_average is None
                else gpu_average >= floor
            ),
        },
    }
    print(
        dumps(
            "JobMetrics",
            data,
            meta={
                "interval_seconds": args.interval,
                "samples": max(
                    summary["samples"]
                    for summary in utilization.values()
                ),
            },
        )
    )
    return 0


def _terminal_hyperlink(label: str, target: str) -> str:
    """Render an OSC 8 link without allowing terminal-control injection."""

    if any(ord(character) < 32 or ord(character) == 127 for character in target):
        raise CoderError("Coder returned an unsafe application URL")
    if not sys.stdout.isatty():
        return target
    return f"\033]8;;{target}\033\\{label}\033]8;;\033\\"


def _interactive_coder_login(url: str) -> str:
    login_url = f"{url.rstrip('/')}/cli-auth"
    print("Coder authentication is required.")
    print("Open this login page, sign in, and copy the session token:")
    print(f"  {_terminal_hyperlink('Open Coder login', login_url)}")
    try:
        token = getpass.getpass("Paste Coder session token: ").strip()
    except EOFError as exc:
        raise CoderError("no Coder session token was entered") from exc
    if not token:
        raise CoderError("no Coder session token was entered")

    # Never persist an unverified credential. A bad paste therefore leaves any
    # existing Coder CLI session untouched.
    try:
        with CoderClient(url, token) as client:
            user = client.current_user()
    except CoderError as exc:
        if exc.status_code in {401, 403}:
            raise CoderError(
                "Coder rejected that session token; open the login page and try again",
                status_code=exc.status_code,
            ) from exc
        raise
    username = str(user.get("username") or user.get("name") or "current user")
    session_path = save_connection(url, token)
    print(f"Coder login saved for {username}: {session_path}")
    return token


def _kill_command(
    args: argparse.Namespace,
    config: Mapping[str, Any],
) -> int:
    """Delete Coder Jobs through Coder and ordinary Jobs through Kubernetes."""

    targets = list(args.jobs) or [target_job(None)]
    coder_jobs = [target for target in targets if target.startswith("coder-")]
    ordinary_jobs = [target for target in targets if not target.startswith("coder-")]
    deleted_workspaces: List[Tuple[str, str]] = []

    if coder_jobs:
        try:
            url, token = resolve_connection(config)
        except CoderAuthenticationRequired as exc:
            if not sys.stdin.isatty() or not sys.stdout.isatty():
                raise
            url = exc.url
            token = _interactive_coder_login(url)

        with CoderClient(url, token) as client:
            user = client.current_user()
            username = str(user.get("username") or user.get("name") or "")
            if not username:
                raise CoderError("Coder did not return the current username")
            owned_workspaces = client.workspaces()
            workspaces: List[Tuple[str, str, Mapping[str, Any]]] = []
            for job in coder_jobs:
                workspace = client.workspace_for_job(
                    job,
                    username=username,
                    workspaces=owned_workspaces,
                )
                name = validate_workspace_name(str(workspace.get("name") or ""))
                workspaces.append((name, job, workspace))
            for name, job, workspace in workspaces:
                client.delete_workspace(workspace)
                deleted_workspaces.append((name, job))

    if ordinary_jobs:
        kill(_namespace_value(args, config), ordinary_jobs)

    if args.output == "json":
        print(dumps("KillResult", {"jobs": targets, "killed": True}))
    else:
        if deleted_workspaces:
            rendered = ", ".join(
                f"{name} ({job})" for name, job in deleted_workspaces
            )
            print(
                "Deleting "
                f"{len(deleted_workspaces)} Coder workspace(s) through Coder: "
                f"{rendered}"
            )
        if ordinary_jobs:
            print(
                f"Killed {len(ordinary_jobs)} Job(s): "
                f"{', '.join(ordinary_jobs)}"
            )
    return 0


def _coder_command(
    args: argparse.Namespace,
    config: Mapping[str, Any],
) -> int:
    resolved_preset = resolve_preset(args.preset, config) if args.preset else None
    existing_reference = args.preset if args.preset and resolved_preset is None else None
    if existing_reference is not None:
        create_options = any(
            (
                args.cpu,
                args.memory,
                args.name,
                args.template,
                args.parameter,
            )
        )
        if create_options:
            raise CoderError(
                "an existing Coder workspace reference cannot be combined with "
                "--cpu, --memory, --name, --template, or --parameter"
            )
        plan = None
    elif resolved_preset is not None:
        preset_name, gpu_count = resolved_preset
        preset = config.get("presets", {}).get(preset_name, {})
        if not isinstance(preset, Mapping):
            raise CoderError(f"invalid Falcon preset {preset_name!r}")
        gpu_type = canonical_gpu(str(preset.get("gpu_type") or preset_name))
        namespace = str(config.get("cluster", {}).get("namespace") or "")
        nodes = _planning_nodes(config, KubernetesClient(namespace))
        shared_memory_percent = preset.get(
            "shared_memory_percent",
            config.get("resources", {}).get("shared_memory_percent", 15),
        )
        plan = plan_resources(
            nodes,
            preset_name,
            gpu_type,
            gpu_count,
            cpu_override=args.cpu,
            memory_override=args.memory,
            shared_memory_percent=shared_memory_percent,
        )
    else:
        if not args.cpu or not args.memory:
            raise CoderError(
                "falcon coder requires a GPU preset or both --cpu and --memory"
            )
        plan = plan_cpu_resources(args.cpu, args.memory)
    coder_config = config.get("coder", {})
    if not isinstance(coder_config, Mapping):
        raise CoderError("coder must be a YAML mapping")
    timeout = (
        args.timeout
        if args.timeout is not None
        else float(coder_config.get("wait_timeout_seconds", 600))
    )
    if not 1 <= timeout <= 3600:
        raise CoderError("--timeout must be between 1 and 3600 seconds")
    try:
        url, token = resolve_connection(config, url_override=args.url)
    except CoderAuthenticationRequired as exc:
        if not sys.stdin.isatty() or not sys.stdout.isatty():
            raise
        url = exc.url
        token = _interactive_coder_login(url)
    overrides = parse_parameter_overrides(args.parameter)
    requested_template = args.template or coder_config.get("template")
    requested_name = validate_workspace_name(args.name) if args.name else None

    with CoderClient(url, token) as client:
        user = client.current_user()
        username = str(user.get("username") or user.get("name") or "")
        if not username:
            raise CoderError("Coder did not return the current username")
        workspace: Mapping[str, Any]
        if existing_reference is not None:
            if existing_reference.startswith("coder-"):
                workspace = client.workspace_for_job(
                    existing_reference,
                    username=username,
                )
                name = validate_workspace_name(
                    str(workspace.get("name") or "")
                )
            else:
                name = validate_workspace_name(existing_reference)
                workspace = client.workspace(username, name)
            print(f"Connecting to existing Coder workspace {name}...")
        else:
            template = resolve_template(
                client.templates(),
                str(requested_template) if requested_template else None,
            )
            template_id = str(template.get("id") or "")
            template_version_id = str(template.get("active_version_id") or "")
            if not template_id or not template_version_id:
                raise CoderError("the selected Coder template has no active version")
            rich_parameters = client.rich_parameters(template_version_id)
            parameters = build_parameter_values(
                rich_parameters,
                plan,
                configured=(
                    coder_config.get("parameters")
                    if isinstance(coder_config.get("parameters"), Mapping)
                    else None
                ),
                overrides=overrides,
            )

            name = requested_name or generate_workspace_name()
            for attempt in range(10):
                gpu_summary = (
                    f" · GPU {plan.gpu.model}x{plan.gpu.count}"
                    if plan.gpu is not None
                    else ""
                )
                print(
                    f"Creating Coder workspace {name} · CPU {plan.compute.cpu} · "
                    f"RAM {plan.compute.memory}{gpu_summary} · "
                    f"template {template.get('name') or template_id}..."
                )
                try:
                    workspace = client.create_workspace(
                        "me",
                        template_id=template_id,
                        name=name,
                        parameters=parameters,
                    )
                    break
                except CoderError as exc:
                    if requested_name and exc.status_code == 409:
                        print(
                            f"Coder workspace {name} already exists; "
                            "using its current resources."
                        )
                        workspace = client.workspace(username, name)
                        break
                    if exc.status_code != 409 or attempt == 9:
                        raise
                    name = generate_workspace_name()
            else:  # pragma: no cover - the loop either breaks or raises
                raise CoderError("could not allocate a unique Coder workspace name")

        print(f"Waiting for Coder agent {name}...")
        workspace = client.wait_until_ready(
            username,
            name,
            timeout=timeout,
        )
        workspace_url, links = build_access_links(
            workspace,
            url,
            folder=str(Path.cwd()),
        )
        selected = select_access_links(links, args.access)
        interactive = sys.stdout.isatty()
        if interactive and any(link.requires_token for link in selected):
            app_token = client.create_app_token()
            selected = tuple(
                link.with_token(app_token) if link.requires_token else link
                for link in selected
            )

    print(f"Workspace ready: {_terminal_hyperlink(name, workspace_url)}")
    job = workspace_job_name(workspace)
    if job:
        print(f"Kubernetes Job: {job}")
    print("Access:")
    for link in selected:
        if link.requires_token:
            # Piped output and logs must never receive a session credential.
            rendered = f"open from {workspace_url} in a terminal"
        else:
            rendered = _terminal_hyperlink("Open", link.target)
        print(f"  {link.label:<20} {rendered}")
    return 0


def _resource_snapshot(
    args: argparse.Namespace,
    config: Mapping[str, Any],
) -> Tuple[object, ClusterSnapshot]:
    if args.demo:
        collector = DemoCollector(args.demo)
        return collector, collector.collect()
    metrics_url = config.get("cluster", {}).get("kube_state_metrics_url")
    if metrics_url:
        metrics_collector = MetricsClusterCollector(str(metrics_url))
        metrics_snapshot = metrics_collector.collect(force=True)
        if metrics_snapshot.nodes:
            return metrics_collector, metrics_snapshot
        metrics_collector.close()
    namespace = _namespace_value(args, config)
    collector = ClusterCollector(
        KubernetesClient(namespace),
        # A node's request headroom is only correct when every visible
        # namespace contributes its active Pods. An explicit --namespace is a
        # deliberate scoped view; the default resource dashboard is cluster
        # wide and therefore asks the adapter for all namespaces.
        namespace=namespace if args.namespace else "",
    )
    snapshot = collector.collect(force=True)
    return collector, snapshot


def _resources_command(
    args: argparse.Namespace,
    config: Mapping[str, Any],
    config_file: Optional[str] = None,
) -> int:
    if not 1 <= args.limit <= 500:
        raise CliError("--limit must be between 1 and 500")
    if not 1 <= args.consumer_limit <= 500:
        raise CliError("--consumer-limit must be between 1 and 500")
    collector, snapshot = _resource_snapshot(args, config)
    if args.output == "human" and sys.stdout.isatty():
        FalconResourcesApp(
            collector,
            node_filter=args.node,
            gpu_filter=args.gpu,
            color_mode=getattr(args, "color", None),
            initial_view=str(
                config.get("resources", {}).get("last_view", "nodes")
            ),
            persist_view=lambda view: save_resources_view(view, config_file),
        ).run(mouse=True)
        return 0
    nodes = list(snapshot.nodes)
    if args.node:
        nodes = [node for node in nodes if args.node.lower() in node.name.lower()]
    if args.gpu:
        wanted = args.gpu.lower()
        nodes = [
            node for node in nodes if wanted in (node.gpu_model or "").lower()
        ]
    nodes = nodes[: args.limit]
    if snapshot.stale and not snapshot.nodes:
        raise KubernetesError(snapshot.error or "Kubernetes inventory unavailable")
    if args.output == "json":
        consumer_count = sum(len(node.visible_consumers) for node in nodes)
        bounded_nodes = [
            replace(
                node,
                consumers=node.visible_consumers[: args.consumer_limit],
            )
            for node in nodes
        ]
        availability = list(snapshot.gpu_availability.values())
        print(
            dumps(
                "ClusterResources",
                {
                    "summary": {
                        "collected_at": snapshot.collected_at,
                        "stale": snapshot.stale,
                        "error": snapshot.error,
                        "nodes": {
                            "total": snapshot.total_nodes,
                            "schedulable": snapshot.schedulable_nodes,
                        },
                        "jobs": {
                            "running": snapshot.running_jobs,
                            "pending": snapshot.pending_jobs,
                        },
                        "pods": {
                            "running": snapshot.running_pods,
                            "pending": snapshot.pending_pods,
                        },
                        "capacity": snapshot.capacity,
                        "allocatable": snapshot.allocatable,
                        "requested": snapshot.requested,
                        "request_headroom": snapshot.request_headroom,
                        "gpu_availability": availability,
                    },
                    "nodes": bounded_nodes,
                },
                meta={
                    "count": len(nodes),
                    "limit": args.limit,
                    "consumer_limit": args.consumer_limit,
                    "consumers_count": consumer_count,
                    "consumers_returned": sum(
                        len(node.consumers) for node in bounded_nodes
                    ),
                    "stale": snapshot.stale,
                },
            )
        )
        return 0
    print(
        f"Nodes {snapshot.schedulable_nodes}/{snapshot.total_nodes} schedulable · "
        f"CPU free {snapshot.request_headroom.cpu_cores:.1f}/"
        f"{snapshot.allocatable.cpu_cores:.1f} · "
        f"RAM free {_memory_gib(snapshot.request_headroom.memory_bytes)}/"
        f"{_memory_gib(snapshot.allocatable.memory_bytes)}"
        + (" · STALE" if snapshot.stale else "")
    )
    print(
        render_table(
            ("NODE", "SCHEDULABLE", "GPU", "VRAM", "CPU", "MEM", "PODS"),
            (
                (
                    node.name,
                    (
                        "not-ready"
                        if node.ready is False
                        else "unknown"
                        if node.ready is None
                        else "yes"
                        if node.schedulable
                        else "cordoned"
                    ),
                    (
                        f"{node.gpu_model} "
                        f"{node.gpu_free}/{node.allocatable.gpu_count}"
                        if node.allocatable.gpu_count
                        else "-"
                    ),
                    (
                        _memory_gib(node.gpu_memory_bytes_per_device)
                        if node.gpu_memory_bytes_per_device is not None
                        else "-"
                    ),
                    f"{node.request_headroom.cpu_cores:.1f}/"
                    f"{node.allocatable.cpu_cores:.1f}",
                    f"{_memory_gib(node.request_headroom.memory_bytes)}/"
                    f"{_memory_gib(node.allocatable.memory_bytes)}",
                    node.workload_count,
                )
                for node in nodes
            ),
            maximum_widths=(40, 12, 18, 10, 16, 18, 6),
        )
    )
    return 0


def _skills_setup(args: argparse.Namespace) -> int:
    operations = []
    if args.uninstall_skills:
        operations = uninstall_skills(args.uninstall_skills)
    elif args.install_skills:
        operations = install_skills(args.install_skills)
    elif not args.skip_skills and not args.non_interactive:
        detected = detect_agents()
        if detected:
            default = ",".join(detected)
            response = input(
                "Install Falcon skill for detected coding agents "
                f"[{default}] (comma-separated, 'none' to skip): "
            ).strip()
            if response.lower() not in {"none", "no", "skip"}:
                operations = install_skills(response or detected)
    conflict = False
    for operation in operations:
        print(f"Skill {operation.agent}: {operation.status} ({operation.path})")
        if operation.detail:
            print(f"  {operation.detail}", file=sys.stderr)
        conflict = conflict or operation.status in {"conflict", "unmanaged"}
    return EXIT_CONFLICT if conflict else 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    config_arg = _config_argument(raw)
    try:
        config = load_config(config_arg)
        rewritten = _rewrite_shorthand(raw, config)
        if _LAUNCH_SENTINEL in rewritten:
            launch_argv = list(rewritten)
            launch_argv.remove(_LAUNCH_SENTINEL)
            return _submit_command(_launch_parser().parse_args(launch_argv), config)
        if (
            "dashboard" in rewritten
            and any(token in {"--json", "--once"} for token in rewritten)
        ):
            raise CliError(
                "dashboard snapshots were removed; use "
                "'falcon jobs --output json' or 'falcon get JOB --output json'"
            )
        parser = _parser(config)
        args = parser.parse_args(rewritten)
        if not args.command_name:
            parser.print_help()
            return 0
        if args.command_name == "jobs":
            return _jobs_command(args, config)
        if args.command_name == "get":
            return _get_command(args, config)
        if args.command_name == "events":
            return _events_command(args, config)
        if args.command_name == "logs":
            return _logs_command(args, config)
        if args.command_name == "attach":
            return attach(_namespace_value(args, config), args.job)
        if args.command_name == "top":
            return top(_namespace_value(args, config), args.job)
        if args.command_name == "metrics":
            return _metrics_command(args, config)
        if args.command_name == "coder":
            return _coder_command(args, config)
        if args.command_name == "kill":
            return _kill_command(args, config)
        if args.command_name == "clean":
            return clean(_namespace_value(args, config))
        if args.command_name == "dashboard":
            if not sys.stdout.isatty():
                raise CliError(
                    "dashboard requires a terminal; use "
                    "'falcon jobs --output json' in noninteractive processes"
                )
            if args.demo:
                run_dashboard(
                    config,
                    demo_state=args.demo,
                    color_mode=getattr(args, "color", None),
                )
            else:
                run_dashboard(
                    config,
                    str(config["cluster"]["namespace"]),
                    job=args.job,
                    config_file=config_arg,
                    color_mode=getattr(args, "color", None),
                )
            return 0
        if args.command_name == "resources":
            return _resources_command(args, config, config_arg)
        if args.command_name == "setup":
            target, rc = run_setup(
                config_arg,
                force=args.force,
                non_interactive=args.non_interactive,
                install_shell=not args.no_shell,
            )
            print(f"Config: {target}")
            if rc:
                print(f"Completion: {rc}")
            return _skills_setup(args)
        if args.command_name == "completion":
            shell = args.shell or detect_shell()[0]
            print(shell_script(shell, config=config), end="")
            return 0
        if args.command_name == "config":
            print(config_path(config_arg))
            return 0
        return 0
    except KeyboardInterrupt:
        return 130
    except KubernetesError as exc:
        code = EXIT_NOT_FOUND if exc.not_found else EXIT_KUBERNETES
        print(f"falcon: {exc}", file=sys.stderr)
        return code
    except CoderError as exc:
        print(f"falcon: {exc}", file=sys.stderr)
        return EXIT_CODER
    except CliError as exc:
        print(f"falcon: {exc}", file=sys.stderr)
        return exc.code
    except (FileNotFoundError, ValueError, OSError) as exc:
        print(f"falcon: {exc}", file=sys.stderr)
        return EXIT_USAGE


if __name__ == "__main__":
    raise SystemExit(main())
