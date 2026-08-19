# Configuration

Falcon reads YAML from `~/.falconrc`, `FALCON_CONFIG`, or `--config PATH`.
`falcon setup` creates version 1 configuration atomically with mode `0600`.

Core fields:

```yaml
version: 1
cluster:
  namespace: research
  gpu_label: gpu-type
  hostname_label: kubernetes.io/hostname
  kube_state_metrics_url: http://localhost:30080/metrics
runtime:
  image: registry.gitlab.com/hvlabs/teams/ai/container-images/base:ubuntu24.04-cuda13.0.2-runtime-withtools-v1.0.0
  scheduler: kai-scheduler
  image_pull_secrets: [hv-gitlab-registry]
  run_as_user: 1000
  run_as_group: 1000
  supplemental_groups: [1000]
  mount_working_dir: true
  mount_home: false
  home: /home/alice
  volumes:
    - /shared:/shared
  environment: {}
  python_environment: auto
resources:
  shared_memory_percent: 15
  consumer_sort: namespace # namespace, cpu, memory, or gpu
  history_enabled: true
  history_hours: 24
  history_interval_seconds: 5
job:
  backoff_limit: null
  ttl_seconds_after_finished: null
presets:
  h100:
    gpu_type: h100
    minimum_utilization: 75
    max_count: 8
  a6000:
    gpu_type: a6000
    max_count: 2
  2080ti:
    gpu_type: 2080ti
    max_count: 4
  pro6000:
    gpu_type: pro6000
    minimum_utilization: 75
    max_count: 2
dashboard:
  ema_alpha: 0.1
  sort_field: Age
  sort_direction: desc
coder:
  url: https://coder.yoda.hyperverge.org
  template: IDEs
  wait_timeout_seconds: 600
  parameters:
    cpu: null
    cpu_limit: null
    memory: null
    memory_limit: null
    gpu_type: null
    gpu_count: null
```

Volume strings are `HOST_PATH[:MOUNT_PATH]`; paths must be absolute. Dict
entries can also specify `name`, `host_path`, `mount_path`, `type`, and
`read_only`.

Normal execution never depends on shell initialization. Completion adds a
small managed block to bash or zsh startup files, but the executable remains
the installed console script.

Environment precedence is explicit CLI path, configured
`runtime.python_environment`, then current `VIRTUAL_ENV`/`CONDA_PREFIX` when
the value is `auto`. Missing or incomplete environments fail before
submission.

Falcon defaults to this deployment's `kai-scheduler` and derives numeric
UID/GID security fields from the invoking process. Override the scheduler or
identity fields only when the target cluster requires different values; use
`scheduler: null` for the Kubernetes default scheduler.

Falcon uses the local kube-state-metrics endpoint for cluster-wide requested
resource accounting. This is the same data path as the original resource
command and works when a user may inspect metrics but cannot list cluster
Nodes. Set `cluster.kube_state_metrics_url` to your site's endpoint, or to
`null` to use Kubernetes API discovery when your credentials have
cluster-wide Node and Pod read access. Transient metrics failures retain the
last valid resource view and visibly mark it stale.

GPU plans derived automatically from node capacity subtract 1 GiB from the
calculated memory request as a safety buffer. An explicit launch-time `-m`
value replaces that automatic memory calculation. Falcon normalizes generated
memory and shared-memory values to integral Kubernetes byte quantities.

`falcon coder` uses the `IDEs` template by default and auto-detects conventional
`cpu`, `cpu_limit`, `memory`, `memory_limit`, `gpu_type`, and `gpu_count`
rich-parameter names. Set a mapping above only when a template
uses a different name. Null limit mappings are optional because many templates
expose one CPU parameter and one RAM parameter, then apply each value to both
the Kubernetes request and limit. Coder credentials are read from
`CODER_SESSION_TOKEN` or the Coder CLI session file and are never stored in
Falcon configuration. If neither exists, an interactive `falcon coder` run
guides login and saves the validated token to the Coder CLI session file with
owner-only permissions.
