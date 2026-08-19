"""Coder workspace creation and editor-link support for Falcon.

Coder, rather than Falcon, must create the Kubernetes Job because the Coder
control plane owns the workspace build and its one-time agent credential.
This module is deliberately a small REST client instead of a wrapper around a
locally installed ``coder`` executable.
"""

from __future__ import annotations

import os
import re
import secrets
import time
from dataclasses import dataclass
from decimal import ROUND_FLOOR, Decimal
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
from urllib.parse import parse_qsl, quote, urlencode, urlsplit, urlunsplit

import httpx
from unique_names_generator import get_random_name
from unique_names_generator.data import ANIMALS, COLORS

from .models import ResourcePlan
from .quantities import parse_cpu, parse_memory_bytes

SESSION_TOKEN_PLACEHOLDER = "$SESSION_TOKEN"
ENCODED_SESSION_TOKEN_PLACEHOLDER = "%24SESSION_TOKEN"
TERMINAL_BUILD_STATES = frozenset(
    {"failed", "canceled", "canceling", "stopping", "stopped", "deleting", "deleted"}
)
WORKSPACE_NAME = re.compile(r"^[a-z0-9](?:[a-z0-9-]{0,30}[a-z0-9])?$")


class CoderError(RuntimeError):
    """A concise, user-facing Coder API or configuration failure."""

    def __init__(self, message: str, *, status_code: Optional[int] = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class CoderAuthenticationRequired(CoderError):
    """The deployment URL is known, but no session credential is available."""

    def __init__(self, url: str) -> None:
        super().__init__(
            "Coder authentication is required; run this command in an "
            "interactive terminal to log in, set CODER_SESSION_TOKEN, or use "
            "an existing Coder CLI session"
        )
        self.url = url


@dataclass(frozen=True)
class AccessLink:
    label: str
    slug: str
    target: str

    @property
    def requires_token(self) -> bool:
        return (
            SESSION_TOKEN_PLACEHOLDER in self.target
            or ENCODED_SESSION_TOKEN_PLACEHOLDER in self.target
        )

    def with_token(self, token: str) -> "AccessLink":
        target = self.target.replace(SESSION_TOKEN_PLACEHOLDER, token)
        target = target.replace(
            ENCODED_SESSION_TOKEN_PLACEHOLDER,
            quote(token, safe=""),
        )
        return AccessLink(self.label, self.slug, target)


def generate_workspace_name() -> str:
    """Match Coder's dashboard shape: lower-case color-animal-(0..99)."""

    for _ in range(100):
        words = get_random_name(
            combo=[COLORS, ANIMALS],
            separator="-",
            style="lowercase",
        )
        candidate = f"{words}-{secrets.randbelow(100)}"
        if (
            len(candidate) <= 32
            and candidate not in {"new", "create"}
            and WORKSPACE_NAME.fullmatch(candidate)
        ):
            return candidate
    raise CoderError("could not generate a valid Coder workspace name")


def validate_workspace_name(value: str) -> str:
    if (
        len(value) > 32
        or value in {"new", "create"}
        or not WORKSPACE_NAME.fullmatch(value)
    ):
        raise CoderError(
            "Coder workspace names must be at most 32 lower-case letters, "
            "numbers, or hyphens and cannot be new or create"
        )
    return value


def resolve_connection(
    config: Mapping[str, Any],
    *,
    url_override: Optional[str] = None,
) -> Tuple[str, str]:
    """Resolve Coder URL/token using the same environment and files as its CLI."""

    root = Path(
        os.environ.get("CODER_CONFIG_DIR", "~/.config/coderv2")
    ).expanduser()

    def read(name: str) -> str:
        try:
            return (root / name).read_text(encoding="utf-8").strip()
        except (FileNotFoundError, OSError):
            return ""

    coder_config = config.get("coder", {})
    configured_url = (
        coder_config.get("url") if isinstance(coder_config, Mapping) else None
    )
    url = (
        url_override
        or os.environ.get("CODER_URL")
        or read("url")
        or configured_url
        or ""
    ).strip().rstrip("/")
    parsed = urlsplit(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise CoderError(
            "Coder URL is missing or invalid; set coder.url in .falconrc "
            "or CODER_URL"
        )
    token = os.environ.get("CODER_SESSION_TOKEN", "").strip() or read("session")
    if not token:
        raise CoderAuthenticationRequired(url)
    return url, token


def save_connection(url: str, token: str) -> Path:
    """Atomically save a validated login in the Coder CLI's standard files."""

    normalized_url = url.strip().rstrip("/")
    parsed = urlsplit(normalized_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise CoderError("refusing to save an invalid Coder URL")
    normalized_token = token.strip()
    if not normalized_token:
        raise CoderError("refusing to save an empty Coder session token")

    root = Path(
        os.environ.get("CODER_CONFIG_DIR", "~/.config/coderv2")
    ).expanduser()
    temporary: List[Path] = []
    try:
        root.mkdir(mode=0o700, parents=True, exist_ok=True)
        root.chmod(0o700)
        for name, value in (("url", normalized_url), ("session", normalized_token)):
            candidate = root / (
                f".{name}.{os.getpid()}.{secrets.token_hex(6)}.tmp"
            )
            descriptor = os.open(
                candidate,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
            )
            temporary.append(candidate)
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                handle.write(value)
                handle.flush()
                os.fsync(handle.fileno())
        for candidate, name in zip(temporary, ("url", "session")):
            destination = root / name
            os.replace(candidate, destination)
            destination.chmod(0o600)
        temporary.clear()
    except OSError as exc:
        raise CoderError(f"could not save Coder login in {root}: {exc}") from exc
    finally:
        for candidate in temporary:
            try:
                candidate.unlink()
            except FileNotFoundError:
                pass
    return root / "session"


class CoderClient:
    """Minimal client for the stable Coder v2 workspace endpoints."""

    def __init__(
        self,
        url: str,
        token: str,
        *,
        timeout: float = 20.0,
        transport: Optional[httpx.BaseTransport] = None,
    ) -> None:
        self.url = url.rstrip("/")
        self.session_token = token
        self._client = httpx.Client(
            base_url=self.url,
            headers={
                "Accept": "application/json",
                "Coder-Session-Token": token,
                "User-Agent": "falcon-k8s/coder",
            },
            timeout=timeout,
            transport=transport,
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "CoderClient":
        return self

    def __exit__(self, *_) -> None:
        self.close()

    def _request(
        self,
        method: str,
        path: str,
        *,
        payload: Optional[Mapping[str, Any]] = None,
        expected: Sequence[int] = (200,),
    ) -> Any:
        try:
            response = self._client.request(method, path, json=payload)
        except httpx.HTTPError as exc:
            raise CoderError(f"could not reach Coder at {self.url}: {exc}") from exc
        if response.status_code not in expected:
            message = response.reason_phrase or f"HTTP {response.status_code}"
            try:
                body = response.json()
            except ValueError:
                body = None
            if isinstance(body, Mapping):
                summary = str(body.get("message") or message)
                detail = str(body.get("detail") or "").strip()
                validations = body.get("validations")
                validation_text = "; ".join(
                    str(item.get("detail") or item.get("field") or "")
                    for item in validations
                    if isinstance(item, Mapping)
                ) if isinstance(validations, list) else ""
                parts = [part for part in (summary, detail, validation_text) if part]
                message = ": ".join(parts)
            raise CoderError(message, status_code=response.status_code)
        if response.status_code == 204 or not response.content:
            return None
        try:
            return response.json()
        except ValueError as exc:
            raise CoderError("Coder returned an invalid JSON response") from exc

    def current_user(self) -> Mapping[str, Any]:
        value = self._request("GET", "/api/v2/users/me")
        if not isinstance(value, Mapping):
            raise CoderError("Coder returned an invalid user response")
        return value

    def workspaces(
        self,
        *,
        query: str = "owner:me",
    ) -> Tuple[Mapping[str, Any], ...]:
        """List matching workspaces, following Coder's offset pagination."""

        result: List[Mapping[str, Any]] = []
        offset = 0
        limit = 100
        while True:
            parameters = urlencode(
                {"q": query, "limit": limit, "offset": offset}
            )
            value = self._request("GET", f"/api/v2/workspaces?{parameters}")
            if not isinstance(value, Mapping):
                raise CoderError("Coder returned an invalid workspace list")
            page = value.get("workspaces", [])
            if not isinstance(page, list):
                raise CoderError("Coder returned an invalid workspace list")
            items = [item for item in page if isinstance(item, Mapping)]
            result.extend(items)
            try:
                count = int(value.get("count", len(result)))
            except (TypeError, ValueError):
                count = len(result)
            if not page or len(result) >= count or len(page) < limit:
                return tuple(result)
            offset += len(page)

    def templates(self) -> Tuple[Mapping[str, Any], ...]:
        value = self._request("GET", "/api/v2/templates")
        if isinstance(value, Mapping):
            value = value.get("templates", [])
        if not isinstance(value, list):
            raise CoderError("Coder returned an invalid template response")
        return tuple(item for item in value if isinstance(item, Mapping))

    def rich_parameters(self, template_version_id: str) -> Tuple[Mapping[str, Any], ...]:
        value = self._request(
            "GET",
            f"/api/v2/templateversions/{quote(template_version_id, safe='')}/rich-parameters",
        )
        if not isinstance(value, list):
            raise CoderError("Coder returned invalid template parameters")
        return tuple(item for item in value if isinstance(item, Mapping))

    def create_workspace(
        self,
        user: str,
        *,
        template_id: str,
        name: str,
        parameters: Sequence[Mapping[str, str]],
    ) -> Mapping[str, Any]:
        value = self._request(
            "POST",
            f"/api/v2/users/{quote(user, safe='')}/workspaces",
            payload={
                "template_id": template_id,
                "name": name,
                "rich_parameter_values": list(parameters),
                "automatic_updates": "never",
            },
            expected=(201,),
        )
        if not isinstance(value, Mapping):
            raise CoderError("Coder returned an invalid workspace response")
        return value

    def workspace(self, user: str, name: str) -> Mapping[str, Any]:
        value = self._request(
            "GET",
            f"/api/v2/users/{quote(user, safe='')}/workspace/{quote(name, safe='')}",
        )
        if not isinstance(value, Mapping):
            raise CoderError("Coder returned an invalid workspace response")
        return value

    def workspace_for_job(
        self,
        job: str,
        *,
        username: str = "",
        workspaces: Optional[Sequence[Mapping[str, Any]]] = None,
    ) -> Mapping[str, Any]:
        """Resolve a template-owned Job without assuming its owner-name format."""

        prefix = f"coder-{username}-" if username else ""
        if workspaces is None and prefix and job.startswith(prefix):
            name = validate_workspace_name(job[len(prefix) :])
            return self.workspace(username, name)

        owned = tuple(workspaces) if workspaces is not None else self.workspaces()
        exact = [item for item in owned if workspace_job_name(item) == job]
        if len(exact) == 1:
            return exact[0]

        # Some templates report only their Terraform resource name to Coder,
        # so the real Kubernetes Job name is not present in the API response.
        # Workspace names are unique per owner; accept only an unambiguous
        # suffix match rather than guessing the template's owner slug.
        suffix_matches = [
            item
            for item in owned
            if str(item.get("name") or "")
            and job.startswith("coder-")
            and job.endswith(f"-{item.get('name')}")
        ]
        if len(suffix_matches) == 1:
            return suffix_matches[0]
        if len(exact) > 1 or len(suffix_matches) > 1:
            raise CoderError(f"Coder Job {job!r} matches multiple workspaces")
        raise CoderError(
            f"Coder Job {job!r} was not found among your Coder workspaces"
        )

    def create_workspace_build(
        self,
        workspace: Mapping[str, Any],
        transition: str,
    ) -> Mapping[str, Any]:
        workspace_id = str(workspace.get("id") or "")
        if not workspace_id:
            raise CoderError("Coder workspace response is missing its ID")
        if transition not in {"start", "stop", "delete"}:
            raise CoderError(f"unsupported Coder workspace transition {transition!r}")
        value = self._request(
            "POST",
            f"/api/v2/workspaces/{quote(workspace_id, safe='')}/builds",
            payload={"transition": transition},
            expected=(200, 201),
        )
        if not isinstance(value, Mapping):
            raise CoderError("Coder returned an invalid workspace build response")
        return value

    def delete_workspace(
        self,
        workspace: Mapping[str, Any],
        *,
        timeout: Optional[float] = None,
        interval: float = 2.0,
    ) -> Mapping[str, Any]:
        """Submit Coder's native delete build for one workspace.

        Deleting the Kubernetes Job directly leaves Coder's Terraform state and
        workspace record behind. A delete build lets Coder destroy the template
        resources and update its own state as one lifecycle action.
        """

        build = self.create_workspace_build(workspace, "delete")
        if timeout is not None:
            self.wait_until_deleted(
                workspace,
                timeout=timeout,
                interval=interval,
            )
        return build

    def wait_until_deleted(
        self,
        workspace: Mapping[str, Any],
        *,
        timeout: float,
        interval: float = 2.0,
    ) -> None:
        """Wait for a Coder delete build to finish, surfacing build failures."""

        workspace_id = str(workspace.get("id") or "")
        name = str(workspace.get("name") or workspace_id)
        if not workspace_id:
            raise CoderError("Coder workspace response is missing its ID")
        deadline = time.monotonic() + timeout
        path = f"/api/v2/workspaces/{quote(workspace_id, safe='')}"
        while True:
            try:
                current = self._request("GET", path)
            except CoderError as exc:
                if exc.status_code == 404:
                    return
                raise
            if not isinstance(current, Mapping):
                raise CoderError("Coder returned an invalid workspace response")
            build = current.get("latest_build", {})
            status = str(build.get("status") or "").lower()
            if status == "deleted":
                return
            if status in {"failed", "canceled"}:
                job = build.get("job", {})
                detail = job.get("error") if isinstance(job, Mapping) else None
                raise CoderError(
                    f"Coder workspace delete {status}"
                    + (f": {detail}" if detail else "")
                )
            if time.monotonic() >= deadline:
                raise CoderError(
                    f"timed out after {timeout:g}s deleting workspace {name}; "
                    f"its current build status is {status or 'unknown'}"
                )
            time.sleep(min(interval, max(0.0, deadline - time.monotonic())))

    def restart_workspace(
        self,
        user: str,
        name: str,
        *,
        timeout: float,
        interval: float = 2.0,
    ) -> Mapping[str, Any]:
        """Restart a workspace with Coder-native stop and start builds."""

        deadline = time.monotonic() + timeout
        workspace = self.workspace(user, name)
        build = workspace.get("latest_build", {})
        status = str(build.get("status") or "").lower()
        if status in {"deleting", "deleted"}:
            raise CoderError(f"Coder workspace {name} is being deleted")

        if status != "stopped":
            if status != "stopping":
                self.create_workspace_build(workspace, "stop")
            while True:
                workspace = self.workspace(user, name)
                build = workspace.get("latest_build", {})
                status = str(build.get("status") or "").lower()
                if status == "stopped":
                    break
                if status in {"failed", "canceled", "deleted"}:
                    job = build.get("job", {})
                    detail = job.get("error") if isinstance(job, Mapping) else None
                    raise CoderError(
                        f"Coder workspace stop {status}"
                        + (f": {detail}" if detail else "")
                    )
                if time.monotonic() >= deadline:
                    raise CoderError(
                        f"timed out after {timeout:g}s stopping workspace {name}"
                    )
                time.sleep(min(interval, max(0.0, deadline - time.monotonic())))

        self.create_workspace_build(workspace, "start")
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise CoderError(
                f"timed out after {timeout:g}s restarting workspace {name}"
            )
        return self.wait_until_ready(
            user,
            name,
            timeout=remaining,
            interval=interval,
        )

    def create_app_token(self) -> str:
        value = self._request(
            "POST", "/api/v2/users/me/keys", payload={}, expected=(201,)
        )
        token = value.get("key") if isinstance(value, Mapping) else None
        if not isinstance(token, str) or not token:
            raise CoderError("Coder did not return an editor application key")
        return token

    def wait_until_ready(
        self,
        user: str,
        name: str,
        *,
        timeout: float,
        interval: float = 2.0,
    ) -> Mapping[str, Any]:
        deadline = time.monotonic() + timeout
        while True:
            workspace = self.workspace(user, name)
            build = workspace.get("latest_build", {})
            status = str(build.get("status") or "").lower()
            if status in TERMINAL_BUILD_STATES:
                job = build.get("job", {})
                detail = job.get("error") if isinstance(job, Mapping) else None
                raise CoderError(
                    f"Coder workspace build {status}"
                    + (f": {detail}" if detail else "")
                )
            agents = workspace_agents(workspace)
            if status == "running" and any(
                str(agent.get("status") or "").lower() == "connected"
                for agent in agents
            ):
                return workspace
            disconnected_off = [
                agent
                for agent in agents
                if str(agent.get("status") or "").lower() == "disconnected"
                and str(agent.get("lifecycle_state") or "").lower() == "off"
            ]
            if status == "running" and agents and len(disconnected_off) == len(agents):
                names = ", ".join(
                    str(agent.get("name") or "unnamed") for agent in disconnected_off
                )
                raise CoderError(
                    f"Coder workspace {name} has no running agent "
                    f"({names}: disconnected, lifecycle off); restart the "
                    "workspace in Coder and try again"
                )
            if time.monotonic() >= deadline:
                raise CoderError(
                    f"timed out after {timeout:g}s waiting for workspace {name}; "
                    f"its current build status is {status or 'unknown'}"
                )
            time.sleep(min(interval, max(0.0, deadline - time.monotonic())))


def resolve_template(
    templates: Sequence[Mapping[str, Any]], requested: Optional[str]
) -> Mapping[str, Any]:
    usable = [item for item in templates if not item.get("deprecated")]
    if requested:
        matches = [
            item
            for item in usable
            if str(item.get("name") or "").casefold() == requested.casefold()
            or str(item.get("display_name") or "").casefold() == requested.casefold()
            or str(item.get("id") or "") == requested
        ]
        if len(matches) == 1:
            return matches[0]
        available = ", ".join(sorted(str(item.get("name")) for item in usable))
        raise CoderError(
            f"Coder template {requested!r} was not found"
            + (f"; available templates: {available}" if available else "")
        )
    if len(usable) == 1:
        return usable[0]
    available = ", ".join(sorted(str(item.get("name")) for item in usable))
    raise CoderError(
        "choose a Coder template with --template or coder.template in .falconrc"
        + (f"; available templates: {available}" if available else "")
    )


PARAMETER_ALIASES: Mapping[str, Tuple[str, ...]] = {
    "cpu": ("cpu", "cpu_request", "cpu_requests", "cpu_cores", "cores"),
    "cpu_limit": ("cpu_limit", "cpu_limits", "limit_cpu"),
    "memory": (
        "memory", "memory_request", "memory_requests", "memory_gib",
        "ram", "ram_request",
    ),
    "memory_limit": ("memory_limit", "memory_limits", "limit_memory", "ram_limit"),
    "gpu_type": ("gpu_type", "gpu_model", "gpu", "accelerator_type"),
    "gpu_count": ("gpu_count", "gpus", "num_gpus", "accelerator_count"),
}


def parse_parameter_overrides(values: Iterable[str]) -> Dict[str, str]:
    result: Dict[str, str] = {}
    for value in values:
        name, separator, setting = value.partition("=")
        if not separator or not name.strip():
            raise CoderError(f"invalid --parameter {value!r}; expected NAME=VALUE")
        result[name.strip()] = setting
    return result


def build_parameter_values(
    parameters: Sequence[Mapping[str, Any]],
    plan: ResourcePlan,
    *,
    configured: Optional[Mapping[str, Any]] = None,
    overrides: Optional[Mapping[str, str]] = None,
) -> Tuple[Mapping[str, str], ...]:
    """Map Falcon request quantities onto a template's rich parameters."""

    configured = configured or {}
    overrides = overrides or {}
    by_name = {
        str(parameter.get("name")): parameter
        for parameter in parameters
        if parameter.get("name")
    }
    canonical = {_canonical(name): value for name, value in by_name.items()}

    def resolve(kind: str, required: bool) -> Optional[Mapping[str, Any]]:
        explicit = configured.get(kind)
        if explicit is not None:
            parameter = by_name.get(str(explicit)) or canonical.get(
                _canonical(str(explicit))
            )
            if parameter is None:
                raise CoderError(
                    f"configured coder.parameters.{kind}={explicit!r} is not "
                    f"present in the template; available parameters: "
                    f"{', '.join(sorted(by_name)) or 'none'}"
                )
            return parameter
        for alias in PARAMETER_ALIASES[kind]:
            parameter = canonical.get(_canonical(alias))
            if parameter is not None:
                return parameter
        if required:
            raise CoderError(
                f"could not identify the template's {kind.replace('_', ' ')} "
                f"parameter; set coder.parameters.{kind} in .falconrc. "
                f"Available parameters: {', '.join(sorted(by_name)) or 'none'}"
            )
        return None

    compute = plan.compute
    resolved = {
        "cpu": resolve("cpu", True),
        "cpu_limit": resolve("cpu_limit", False),
        "memory": resolve("memory", True),
        "memory_limit": resolve("memory_limit", False),
    }
    if plan.gpu is not None:
        resolved["gpu_type"] = resolve("gpu_type", True)
        resolved["gpu_count"] = resolve("gpu_count", True)
    output: Dict[str, str] = {}
    for kind, parameter in resolved.items():
        if parameter is None:
            continue
        if kind == "gpu_type":
            output[str(parameter["name"])] = _option_value(
                parameter, plan.gpu.model
            )
            continue
        if kind == "gpu_count":
            output[str(parameter["name"])] = _option_value(
                parameter, str(plan.gpu.count)
            )
            continue
        quantity = (
            compute.memory if kind.startswith("memory") else compute.cpu
        )
        output[str(parameter["name"])] = _parameter_value(
            parameter, quantity, memory=kind.startswith("memory")
        )
    for name, value in overrides.items():
        parameter = by_name.get(name) or canonical.get(_canonical(name))
        if parameter is None:
            raise CoderError(
                f"template parameter {name!r} does not exist; available parameters: "
                f"{', '.join(sorted(by_name)) or 'none'}"
            )
        output[str(parameter["name"])] = value
    return tuple({"name": name, "value": value} for name, value in output.items())


def _option_value(parameter: Mapping[str, Any], value: str) -> str:
    options = parameter.get("options")
    if not isinstance(options, list) or not options:
        return value
    values = {
        str(item.get("value")).casefold(): str(item.get("value"))
        for item in options
        if isinstance(item, Mapping) and item.get("value") is not None
    }
    selected = values.get(value.casefold())
    if selected is None:
        raise CoderError(
            f"template parameter {parameter.get('name')!r} does not accept "
            f"{value!r}; available values: {', '.join(values.values())}"
        )
    return selected


def _parameter_value(
    parameter: Mapping[str, Any], quantity: str, *, memory: bool
) -> str:
    if memory:
        byte_value = parse_memory_bytes(quantity)
        gib_value = byte_value / (Decimal(1024) ** 3)
        numeric_value = gib_value
        numeric = _decimal_text(gib_value)
        candidates = (quantity, numeric, _decimal_text(byte_value))
    else:
        numeric_value = parse_cpu(quantity)
        numeric = _decimal_text(numeric_value)
        candidates = (numeric, quantity)
    options = parameter.get("options")
    if isinstance(options, list):
        option_values = {
            str(item.get("value")): str(item.get("value"))
            for item in options
            if isinstance(item, Mapping) and item.get("value") is not None
        }
        folded = {key.casefold(): value for key, value in option_values.items()}
        for candidate in candidates:
            if candidate in option_values:
                return option_values[candidate]
            if candidate.casefold() in folded:
                return folded[candidate.casefold()]
    if str(parameter.get("type") or "").lower() == "number":
        return _integer_parameter_value(parameter, numeric_value)
    return quantity


def _integer_parameter_value(
    parameter: Mapping[str, Any], value: Decimal
) -> str:
    """Convert a resource quantity to Coder's integer ``number`` type."""

    integer = value.to_integral_value(rounding=ROUND_FLOOR)
    name = str(parameter.get("name") or "number")
    if integer <= 0:
        raise CoderError(
            f"template parameter {name!r} only accepts whole numbers; "
            f"the resolved value {_decimal_text(value)!r} is below 1"
        )
    minimum = parameter.get("validation_min")
    maximum = parameter.get("validation_max")
    if minimum is not None and integer < Decimal(str(minimum)):
        raise CoderError(
            f"template parameter {name!r} requires at least {minimum}; "
            f"Falcon resolved {_decimal_text(integer)}"
        )
    if maximum is not None and integer > Decimal(str(maximum)):
        raise CoderError(
            f"template parameter {name!r} allows at most {maximum}; "
            f"Falcon resolved {_decimal_text(integer)}"
        )
    return _decimal_text(integer)


def _decimal_text(value: Decimal) -> str:
    text = format(value, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text or "0"


def _canonical(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.casefold())


def _open_folder(target: str, folder: str) -> str:
    """Set an editor URI's remote folder while preserving its auth values."""

    parsed = urlsplit(target)
    query = [
        (key, value)
        for key, value in parse_qsl(parsed.query, keep_blank_values=True)
        if key.casefold() not in {"folder", "openrecent"}
    ]
    query.append(("folder", folder))
    return urlunsplit(parsed._replace(query=urlencode(query)))


def _workspace_app_url(
    base: str,
    owner: str,
    workspace: str,
    app: Mapping[str, Any],
) -> str:
    """Build Coder's authenticated public URL for an internal workspace app."""

    slug = str(app.get("slug") or "").strip()
    if not slug:
        return ""
    subdomain_name = str(app.get("subdomain_name") or "").strip()
    if app.get("subdomain") and subdomain_name:
        parsed = urlsplit(base)
        if not parsed.hostname:
            return ""
        host = f"{subdomain_name}.{parsed.hostname}"
        if parsed.port is not None:
            host = f"{host}:{parsed.port}"
        return urlunsplit((parsed.scheme, host, "/", "", ""))
    return (
        f"{base}/@{quote(owner, safe='')}/{quote(workspace, safe='')}/apps/"
        f"{quote(slug, safe='-')}/"
    )


def workspace_agents(workspace: Mapping[str, Any]) -> Tuple[Mapping[str, Any], ...]:
    build = workspace.get("latest_build", {})
    resources = build.get("resources", []) if isinstance(build, Mapping) else []
    agents: List[Mapping[str, Any]] = []
    if isinstance(resources, list):
        for resource in resources:
            if not isinstance(resource, Mapping):
                continue
            values = resource.get("agents", [])
            if isinstance(values, list):
                agents.extend(item for item in values if isinstance(item, Mapping))
    return tuple(agents)


def workspace_job_name(workspace: Mapping[str, Any]) -> str:
    build = workspace.get("latest_build", {})
    resources = build.get("resources", []) if isinstance(build, Mapping) else []
    if isinstance(resources, list):
        for resource in resources:
            if not isinstance(resource, Mapping):
                continue
            name = str(resource.get("name") or "")
            if name.startswith("coder-"):
                return name
    owner = str(workspace.get("owner_name") or "")
    name = str(workspace.get("name") or "")
    return f"coder-{owner}-{name}" if owner and name else ""


def build_access_links(
    workspace: Mapping[str, Any],
    access_url: str,
    *,
    folder: Optional[str] = None,
) -> Tuple[str, Tuple[AccessLink, ...]]:
    """Build Coder-supported editor links, optionally rooted at one folder."""

    if folder is not None and not Path(folder).is_absolute():
        raise CoderError("the Coder editor folder must be an absolute path")

    owner = str(workspace.get("owner_name") or "")
    name = str(workspace.get("name") or "")
    if not owner or not name:
        raise CoderError("Coder workspace response is missing its owner or name")
    base = access_url.rstrip("/")
    workspace_url = f"{base}/@{quote(owner, safe='')}/{quote(name, safe='')}"
    agents = workspace_agents(workspace)
    if not agents:
        return workspace_url, ()
    agent = next(
        (
            item for item in agents
            if str(item.get("status") or "").lower() == "connected"
        ),
        agents[0],
    )
    agent_name = str(agent.get("name") or "")
    display_apps = {
        str(value).lower() for value in agent.get("display_apps", [])
    } if isinstance(agent.get("display_apps"), list) else set()
    links: List[AccessLink] = []
    if not display_apps or "vscode" in display_apps:
        values = {
            "url": base,
            "owner": owner,
            "workspace": name,
            "agent": agent_name,
            "token": SESSION_TOKEN_PLACEHOLDER,
        }
        if folder is None:
            values["openRecent"] = "true"
        else:
            values["folder"] = folder
        query = urlencode(values)
        links.append(
            AccessLink("VS Code", "vscode", f"vscode://coder.coder-remote/open?{query}")
        )

    apps = agent.get("apps", [])
    if isinstance(apps, list):
        for app in apps:
            if not isinstance(app, Mapping) or app.get("hidden"):
                continue
            label = str(app.get("display_name") or app.get("slug") or "").strip()
            slug = str(app.get("slug") or label).strip()
            wanted = _canonical(f"{label} {slug}")
            if not any(
                value in wanted for value in ("antigravity", "cursor", "jupyter")
            ):
                continue
            # The current IDEs template exposes the modern client generically
            # as "Antigravity IDE"/"antigravity". Canonicalize that server
            # label so the printed choices distinguish its two URI handlers.
            if (
                "antigravity" in wanted
                and "ide" in wanted
                and "20" not in wanted
            ):
                label = "Antigravity 2.0"
                slug = "antigravity-2-0"
                wanted = _canonical(f"{label} {slug}")
            target = (
                str(app.get("url") or "")
                if app.get("external")
                else _workspace_app_url(base, owner, name, app)
            )
            command = str(app.get("command") or "")
            if not target and command:
                qualified = f"{name}.{agent_name}" if agent_name else name
                target = (
                    f"{base}/@{quote(owner, safe='')}/{quote(qualified, safe='.')}/terminal?"
                    + urlencode({"command": command})
                )
            if target:
                if folder is not None and "jupyter" not in wanted:
                    target = _open_folder(target, folder)
                if (
                    "antigravity" in wanted
                    and "20" in wanted
                    and "ide" in wanted
                    and target.partition(":")[0].casefold() == "antigravity"
                ):
                    target = f"antigravity-ide:{target.partition(':')[2]}"
                links.append(AccessLink(label, slug, target))

    has_antigravity_2_ide = any(
        "antigravity" in _canonical(f"{link.label} {link.slug}")
        and "20" in _canonical(f"{link.label} {link.slug}")
        and "ide" in _canonical(f"{link.label} {link.slug}")
        for link in links
    )
    if not has_antigravity_2_ide:
        antigravity_2 = next(
            (
                link
                for link in links
                if "antigravity" in _canonical(f"{link.label} {link.slug}")
                and "20" in _canonical(f"{link.label} {link.slug}")
                and "ide" not in _canonical(f"{link.label} {link.slug}")
                and link.target.partition(":")[0].casefold() == "antigravity"
            ),
            None,
        )
        if antigravity_2 is not None:
            links.append(
                AccessLink(
                    "Antigravity 2.0 IDE",
                    "antigravity-2-0-ide",
                    f"antigravity-ide:{antigravity_2.target.partition(':')[2]}",
                )
            )

    # The generic Antigravity 2.0 URI is only an input for constructing the
    # dedicated IDE handler. Do not print both entries: the IDE link is the
    # supported user-facing choice.
    links = [
        link
        for link in links
        if not (
            "antigravity" in _canonical(f"{link.label} {link.slug}")
            and "20" in _canonical(f"{link.label} {link.slug}")
            and "ide" not in _canonical(f"{link.label} {link.slug}")
        )
    ]

    if not display_apps or "web_terminal" in display_apps:
        qualified = f"{name}.{agent_name}" if agent_name else name
        links.append(
            AccessLink(
                "Terminal",
                "terminal",
                f"{base}/@{quote(owner, safe='')}/{quote(qualified, safe='.')}/terminal",
            )
        )
    links.sort(key=_link_order)
    return workspace_url, tuple(links)


def select_access_links(
    links: Sequence[AccessLink], requested: str
) -> Tuple[AccessLink, ...]:
    if requested.casefold() == "all":
        return tuple(links)
    wanted = _canonical(requested)
    aliases = {
        "antigravity2": "antigravity20",
        "antigravity20": "antigravity20",
        "code": "vscode",
        "jupyter": "jupyterlab",
        "notebook": "jupyterlab",
    }
    wanted = aliases.get(wanted, wanted)
    exact = []
    for link in links:
        identifiers = {
            aliases.get(_canonical(link.label), _canonical(link.label)),
            aliases.get(_canonical(link.slug), _canonical(link.slug)),
        }
        if wanted in identifiers:
            exact.append(link)
    if exact:
        return tuple(exact)
    matches = []
    for link in links:
        candidate = _canonical(f"{link.label} {link.slug}")
        comparable = candidate.replace("antigravity2", "antigravity20")
        if wanted in {"vscode", "cursor", "terminal"}:
            matched = wanted in comparable
        else:
            matched = wanted == comparable or wanted in comparable
        if matched:
            matches.append(link)
    if not matches:
        available = ", ".join(link.label for link in links) or "none"
        raise CoderError(
            f"workspace access option {requested!r} is unavailable; available: {available}"
        )
    return tuple(matches)


def _link_order(link: AccessLink) -> Tuple[int, str]:
    value = _canonical(f"{link.label} {link.slug}")
    if "vscode" in value:
        return 10, value
    if "antigravity" in value and ("20" in value or value.count("2")):
        return 30, value
    if "antigravity" in value:
        return 20, value
    if "cursor" in value:
        return 40, value
    if "jupyter" in value:
        return 45, value
    if "terminal" in value:
        return 50, value
    return 60, value
