"""Bounded Kubernetes operations behind one small argv-only adapter."""

from __future__ import annotations

import json
import posixpath
import subprocess
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence

import yaml

from .models import SubmittedJob

DEFAULT_TIMEOUT = 15.0

_ZSH_DEBUG_RC = """\
export CONDA_AUTO_ACTIVATE_BASE=false
export CONDA_CHANGEPS1=false
typeset -gi FALCON_HAD_CONDA_PREFIX=${+CONDA_PREFIX}
typeset -gi FALCON_HAD_CONDA_DEFAULT_ENV=${+CONDA_DEFAULT_ENV}
typeset -gi FALCON_HAD_CONDA_SHLVL=${+CONDA_SHLVL}
typeset -gi FALCON_HAD_VIRTUAL_ENV=${+VIRTUAL_ENV}
typeset -g FALCON_SAVED_CONDA_PREFIX="${CONDA_PREFIX-}"
typeset -g FALCON_SAVED_CONDA_DEFAULT_ENV="${CONDA_DEFAULT_ENV-}"
typeset -g FALCON_SAVED_CONDA_SHLVL="${CONDA_SHLVL-}"
typeset -g FALCON_SAVED_VIRTUAL_ENV="${VIRTUAL_ENV-}"
if [[ -r "$FALCON_USER_RC" ]]; then
  export ZDOTDIR="${FALCON_USER_RC:h}"
  source "$FALCON_USER_RC"
fi
if (( FALCON_HAD_CONDA_PREFIX )); then
  export CONDA_PREFIX="$FALCON_SAVED_CONDA_PREFIX"
  export PATH="$FALCON_SAVED_CONDA_PREFIX/bin:$PATH"
else
  unset CONDA_PREFIX
fi
if (( FALCON_HAD_CONDA_DEFAULT_ENV )); then
  export CONDA_DEFAULT_ENV="$FALCON_SAVED_CONDA_DEFAULT_ENV"
else
  unset CONDA_DEFAULT_ENV
fi
if (( FALCON_HAD_CONDA_SHLVL )); then
  export CONDA_SHLVL="$FALCON_SAVED_CONDA_SHLVL"
else
  unset CONDA_SHLVL
fi
if (( FALCON_HAD_VIRTUAL_ENV )); then
  export VIRTUAL_ENV="$FALCON_SAVED_VIRTUAL_ENV"
  export PATH="$FALCON_SAVED_VIRTUAL_ENV/bin:$PATH"
else
  unset VIRTUAL_ENV
fi
function _falcon_prompt_prefix {
  if [[ "$PROMPT" != "(${FALCON_PROMPT_LABEL}) "* ]]; then
    PROMPT="(${FALCON_PROMPT_LABEL}) ${PROMPT:-%~ %# }"
  fi
}
autoload -Uz add-zsh-hook
add-zsh-hook precmd _falcon_prompt_prefix
_falcon_prompt_prefix
unset FALCON_SAVED_CONDA_PREFIX
unset FALCON_SAVED_CONDA_DEFAULT_ENV FALCON_SAVED_CONDA_SHLVL
unset FALCON_SAVED_VIRTUAL_ENV
"""

_BASH_DEBUG_RC = """\
export CONDA_AUTO_ACTIVATE_BASE=false
export CONDA_CHANGEPS1=false
FALCON_HAD_CONDA_PREFIX=${CONDA_PREFIX+x}
FALCON_HAD_CONDA_DEFAULT_ENV=${CONDA_DEFAULT_ENV+x}
FALCON_HAD_CONDA_SHLVL=${CONDA_SHLVL+x}
FALCON_HAD_VIRTUAL_ENV=${VIRTUAL_ENV+x}
FALCON_SAVED_CONDA_PREFIX=${CONDA_PREFIX-}
FALCON_SAVED_CONDA_DEFAULT_ENV=${CONDA_DEFAULT_ENV-}
FALCON_SAVED_CONDA_SHLVL=${CONDA_SHLVL-}
FALCON_SAVED_VIRTUAL_ENV=${VIRTUAL_ENV-}
if [[ -r "$FALCON_USER_RC" ]]; then
  source "$FALCON_USER_RC"
fi
if [[ $FALCON_HAD_CONDA_PREFIX == x ]]; then
  export CONDA_PREFIX="$FALCON_SAVED_CONDA_PREFIX"
  export PATH="$FALCON_SAVED_CONDA_PREFIX/bin:$PATH"
else
  unset CONDA_PREFIX
fi
if [[ $FALCON_HAD_CONDA_DEFAULT_ENV == x ]]; then
  export CONDA_DEFAULT_ENV="$FALCON_SAVED_CONDA_DEFAULT_ENV"
else
  unset CONDA_DEFAULT_ENV
fi
if [[ $FALCON_HAD_CONDA_SHLVL == x ]]; then
  export CONDA_SHLVL="$FALCON_SAVED_CONDA_SHLVL"
else
  unset CONDA_SHLVL
fi
if [[ $FALCON_HAD_VIRTUAL_ENV == x ]]; then
  export VIRTUAL_ENV="$FALCON_SAVED_VIRTUAL_ENV"
  export PATH="$FALCON_SAVED_VIRTUAL_ENV/bin:$PATH"
else
  unset VIRTUAL_ENV
fi
__falcon_prompt_prefix() {
  if [[ "$PS1" != "(${FALCON_PROMPT_LABEL}) "* ]]; then
    PS1="(${FALCON_PROMPT_LABEL}) ${PS1:-\\w \\$ }"
  fi
}
if [[ -n ${PROMPT_COMMAND-} ]]; then
  PROMPT_COMMAND="${PROMPT_COMMAND};__falcon_prompt_prefix"
else
  PROMPT_COMMAND=__falcon_prompt_prefix
fi
__falcon_prompt_prefix
unset FALCON_SAVED_CONDA_PREFIX
unset FALCON_SAVED_CONDA_DEFAULT_ENV FALCON_SAVED_CONDA_SHLVL
unset FALCON_SAVED_VIRTUAL_ENV
"""


class KubernetesError(RuntimeError):
    """A Kubernetes operation failed with a useful process classification."""

    def __init__(
        self,
        message: str,
        *,
        returncode: int = 1,
        stderr: str = "",
        command: Sequence[str] = (),
    ) -> None:
        super().__init__(message)
        self.returncode = returncode
        self.stderr = stderr
        self.command = tuple(command)

    @property
    def not_found(self) -> bool:
        text = f"{self} {self.stderr}".lower()
        return "notfound" in text or "not found" in text

    @property
    def unavailable(self) -> bool:
        text = f"{self} {self.stderr}".lower()
        markers = (
            "unable to connect",
            "connection refused",
            "i/o timeout",
            "context deadline exceeded",
            "no such host",
            "couldn't get current server api group",
        )
        return any(marker in text for marker in markers)


@dataclass(frozen=True)
class ProcessResult:
    argv: tuple[str, ...]
    returncode: int
    stdout: str = ""
    stderr: str = ""


class KubernetesClient:
    """Minimal kubectl adapter used by CLI commands and collectors.

    All calls are direct argv executions.  Read operations have bounded
    timeouts, and callers receive parsed dictionaries rather than human prose.
    """

    def __init__(
        self,
        namespace: str = "default",
        *,
        executable: str = "kubectl",
        timeout: float = DEFAULT_TIMEOUT,
        runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
        clock: Callable[[], float] = time.monotonic,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        if not namespace:
            raise ValueError("Kubernetes namespace must not be empty")
        if timeout <= 0:
            raise ValueError("Kubernetes timeout must be positive")
        self.namespace = namespace
        self.executable = executable
        self.timeout = float(timeout)
        self._runner = runner
        self._clock = clock
        self._sleep = sleeper
        self._closed = False

    def _run(
        self,
        args: Sequence[str],
        *,
        input_text: Optional[str] = None,
        timeout: Optional[float] = None,
        capture: bool = True,
        check: bool = True,
    ) -> ProcessResult:
        if self._closed:
            raise KubernetesError("Kubernetes client is closed")
        argv = [self.executable, *map(str, args)]
        timeout_value = (
            self.timeout
            if timeout is None
            else (None if timeout <= 0 else timeout)
        )
        try:
            completed = self._runner(
                argv,
                input=input_text,
                capture_output=capture,
                text=True,
                timeout=timeout_value,
                check=False,
            )
        except FileNotFoundError as exc:
            raise KubernetesError(
                f"{self.executable!r} was not found on PATH",
                command=argv,
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise KubernetesError(
                f"Kubernetes command timed out after {exc.timeout}s",
                command=argv,
            ) from exc
        except OSError as exc:
            raise KubernetesError(
                f"could not start Kubernetes command: {exc}",
                command=argv,
            ) from exc
        result = ProcessResult(
            argv=tuple(argv),
            returncode=int(completed.returncode),
            stdout=completed.stdout or "",
            stderr=completed.stderr or "",
        )
        if check and result.returncode:
            detail = result.stderr.strip() or result.stdout.strip()
            raise KubernetesError(
                detail or f"Kubernetes command exited {result.returncode}",
                returncode=result.returncode,
                stderr=result.stderr,
                command=argv,
            )
        return result

    @staticmethod
    def _json(result: ProcessResult) -> Dict[str, Any]:
        try:
            value = json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            raise KubernetesError(
                "Kubernetes returned invalid JSON",
                returncode=result.returncode,
                stderr=result.stderr,
                command=result.argv,
            ) from exc
        if not isinstance(value, dict):
            raise KubernetesError(
                "Kubernetes returned a non-object JSON value",
                command=result.argv,
            )
        return value

    def list_json(
        self,
        resource: str,
        *,
        namespace: Optional[str] = None,
        all_namespaces: bool = False,
        labels: Optional[Mapping[str, str]] = None,
        field_selector: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> Dict[str, Any]:
        args = ["get", resource]
        if all_namespaces:
            args.append("--all-namespaces")
        elif namespace is not None or resource.lower() not in {"node", "nodes"}:
            args += ["--namespace", namespace or self.namespace]
        if labels:
            selector = ",".join(
                f"{key}={value}" for key, value in sorted(labels.items())
            )
            args += ["--selector", selector]
        if field_selector:
            args += ["--field-selector", field_selector]
        if limit is not None:
            if limit <= 0:
                raise ValueError("Kubernetes list limit must be positive")
            args += ["--chunk-size", str(limit)]
        args += ["--output", "json"]
        return self._json(self._run(args))

    def get_json(
        self,
        resource: str,
        name: str,
        *,
        namespace: Optional[str] = None,
    ) -> Dict[str, Any]:
        if not name:
            raise ValueError("Kubernetes object name must not be empty")
        args = ["get", resource, name]
        if resource.lower() not in {"node", "nodes"}:
            args += ["--namespace", namespace or self.namespace]
        args += ["--output", "json"]
        return self._json(self._run(args))

    def list_nodes(self) -> Dict[str, Any]:
        return self.list_json("nodes")

    def list_pods(self, namespace: Optional[str] = None) -> Dict[str, Any]:
        return self.list_json(
            "pods",
            namespace=namespace or self.namespace,
            all_namespaces=namespace == "",
        )

    def list_jobs(self, namespace: Optional[str] = None) -> Dict[str, Any]:
        return self.list_json(
            "jobs.batch",
            namespace=namespace or self.namespace,
            all_namespaces=namespace == "",
        )

    def list_inventory(self, namespace: Optional[str] = None) -> Dict[str, Any]:
        """Fetch each slow inventory source once per collector refresh."""
        effective = self.namespace if namespace is None else namespace
        nodes = self.list_nodes()
        pods = self.list_pods(effective)
        jobs = self.list_jobs(effective)
        return {"nodes": nodes, "pods": pods, "jobs": jobs}

    def create_job(self, manifest: Mapping[str, Any]) -> SubmittedJob:
        payload = yaml.safe_dump(dict(manifest), sort_keys=False)
        created = self._json(
            self._run(
                ["create", "--filename", "-", "--output", "json"],
                input_text=payload,
                timeout=max(self.timeout, 30.0),
            )
        )
        metadata = created.get("metadata") or {}
        return SubmittedJob(
            name=str(metadata.get("name") or ""),
            namespace=str(metadata.get("namespace") or self.namespace),
            uid=str(metadata.get("uid")) if metadata.get("uid") else None,
            resource_version=(
                str(metadata.get("resourceVersion"))
                if metadata.get("resourceVersion")
                else None
            ),
            created=True,
        )

    def delete_jobs(
        self,
        names: Iterable[str],
        *,
        namespace: Optional[str] = None,
        wait: bool = False,
    ) -> ProcessResult:
        targets = [name for name in names if name]
        if not targets:
            raise ValueError("at least one Job name is required")
        return self._run(
            [
                "delete", "jobs.batch", *targets,
                "--namespace", namespace or self.namespace,
                f"--wait={'true' if wait else 'false'}",
            ],
            timeout=max(self.timeout, 40.0) if wait else self.timeout,
        )

    def delete_pod(
        self,
        name: str,
        *,
        namespace: Optional[str] = None,
        wait: bool = False,
    ) -> ProcessResult:
        return self._run(
            [
                "delete", "pod", name,
                "--namespace", namespace or self.namespace,
                f"--wait={'true' if wait else 'false'}",
            ]
        )

    def job_pods(
        self,
        name: str,
        *,
        namespace: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        payload = self.list_json(
            "pods",
            namespace=namespace or self.namespace,
            labels={"job-name": name},
        )
        return list(payload.get("items") or [])

    def wait_for_pod(
        self,
        job_name: str,
        *,
        namespace: Optional[str] = None,
        timeout: float = 300.0,
        poll_seconds: float = 1.0,
    ) -> Dict[str, Any]:
        if timeout <= 0 or poll_seconds <= 0:
            raise ValueError("wait timeout and poll interval must be positive")
        deadline = self._clock() + timeout
        last_states: List[str] = []
        while True:
            pods = self.job_pods(job_name, namespace=namespace)
            last_states = [
                str((pod.get("status") or {}).get("phase") or "Unknown")
                for pod in pods
            ]
            running = [
                pod for pod in pods
                if (pod.get("status") or {}).get("phase") == "Running"
                and not (pod.get("metadata") or {}).get("deletionTimestamp")
            ]
            if running:
                return max(
                    running,
                    key=lambda pod: str(
                        (pod.get("metadata") or {}).get(
                            "creationTimestamp", ""
                        )
                    ),
                )
            if pods and all(state in {"Failed", "Succeeded"} for state in last_states):
                raise KubernetesError(
                    f"Job {job_name} has no active Pod "
                    f"(attempt states: {', '.join(last_states)})"
                )
            remaining = deadline - self._clock()
            if remaining <= 0:
                state = ", ".join(last_states) if last_states else "no Pods created"
                raise KubernetesError(
                    f"timed out waiting for Job {job_name} Pod ({state})"
                )
            self._sleep(min(poll_seconds, remaining))

    def logs(
        self,
        job_name: str,
        *,
        namespace: Optional[str] = None,
        tail: int = 100,
        follow: bool = False,
        container: Optional[str] = None,
    ) -> ProcessResult:
        if tail < 0 or tail > 100_000:
            raise ValueError("log tail must be between 0 and 100000")
        args = [
            "logs", f"job.batch/{job_name}",
            "--namespace", namespace or self.namespace,
            "--tail", str(tail),
        ]
        if container:
            args += ["--container", container]
        if follow:
            args += ["--follow", "--pod-running-timeout=5m"]
        return self._run(
            args,
            capture=not follow,
            check=False,
            timeout=0 if follow else self.timeout,
        )

    def attach(
        self,
        job_name: str,
        *,
        namespace: Optional[str] = None,
        timeout: float = 300.0,
    ) -> ProcessResult:
        effective = namespace or self.namespace
        pod = self.wait_for_pod(
            job_name, namespace=effective, timeout=timeout
        )
        pod_name = str((pod.get("metadata") or {}).get("name") or "")
        return self._run(
            ["attach", pod_name, "--namespace", effective, "--stdin", "--tty"],
            capture=False,
            check=False,
            timeout=0,
        )

    def exec_shell(
        self,
        job_name: str,
        shell: str,
        *,
        prompt_label: str = "falcon",
        rc_path: Optional[str] = None,
        namespace: Optional[str] = None,
        timeout: float = 300.0,
    ) -> ProcessResult:
        effective = namespace or self.namespace
        pod = self.wait_for_pod(
            job_name, namespace=effective, timeout=timeout
        )
        pod_name = str((pod.get("metadata") or {}).get("name") or "")
        shell_name = shell.rsplit("/", 1)[-1]
        if shell_name == "zsh":
            wrapper_dir = "/tmp/falcon-zdotdir"
            wrapper_path = posixpath.join(wrapper_dir, ".zshrc")
            wrapper = _ZSH_DEBUG_RC
            shell_args = [shell, "-i"]
        elif shell_name == "bash":
            wrapper_dir = "/tmp/falcon-bash"
            wrapper_path = posixpath.join(wrapper_dir, ".bashrc")
            wrapper = _BASH_DEBUG_RC
            shell_args = [shell, "--noprofile", "--rcfile", wrapper_path, "-i"]
        else:
            raise ValueError(
                f"unsupported interactive shell {shell!r}; expected zsh or bash"
            )
        self._run(
            [
                "exec", "--namespace", effective, pod_name,
                "--", "mkdir", "-p", wrapper_dir,
            ]
        )
        self._run(
            [
                "exec", "--stdin", "--namespace", effective, pod_name,
                "--", "tee", wrapper_path,
            ],
            input_text=wrapper,
        )
        environment = [
            "env",
            f"FALCON_PROMPT_LABEL={prompt_label}",
            f"FALCON_USER_RC={rc_path or ''}",
            "CONDA_AUTO_ACTIVATE_BASE=false",
            "CONDA_CHANGEPS1=false",
        ]
        if shell_name == "zsh":
            environment.append(f"ZDOTDIR={wrapper_dir}")
        return self._run(
            [
                "exec", "--stdin", "--tty",
                "--namespace", effective, pod_name,
                "--", *environment, *shell_args,
            ],
            capture=False,
            check=False,
            timeout=0,
        )

    def top(
        self,
        job_name: str,
        *,
        namespace: Optional[str] = None,
        timeout: float = 60.0,
    ) -> ProcessResult:
        effective = namespace or self.namespace
        pod = self.wait_for_pod(
            job_name, namespace=effective, timeout=timeout
        )
        pod_name = str((pod.get("metadata") or {}).get("name") or "")
        return self._run(
            [
                "exec", "--stdin", "--tty",
                "--namespace", effective, pod_name,
                "--", "python", "-m", "nvitop",
            ],
            capture=False,
            check=False,
            timeout=0,
        )

    def events(
        self,
        names: Iterable[str],
        *,
        namespace: Optional[str] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        if limit <= 0 or limit > 500:
            raise ValueError("event limit must be between 1 and 500")
        wanted = {name for name in names if name}
        payload = self.list_json(
            "events", namespace=namespace or self.namespace
        )
        values = [
            item for item in (payload.get("items") or [])
            if str((item.get("involvedObject") or {}).get("name") or "")
            in wanted
        ]

        def stamp(item: Mapping[str, Any]) -> str:
            metadata = item.get("metadata") or {}
            series = item.get("series") or {}
            return str(
                item.get("eventTime")
                or item.get("lastTimestamp")
                or series.get("lastObservedTime")
                or metadata.get("creationTimestamp")
                or ""
            )

        return sorted(values, key=stamp)[-limit:]

    def close(self) -> None:
        self._closed = True
