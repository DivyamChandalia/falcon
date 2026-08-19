# Resource semantics

Falcon keeps three concepts separate:

- **Requested** comes from the Job Pod template. It survives Pod deletion and
  Job completion.
- **Allocated now** comes only from active node-bound Pods.
- **Observed usage** comes from metrics and may be unavailable.

A completed 2×H100 Job therefore reports:

```text
Status          Succeeded
GPU requested   H100 x2
GPU allocated   -
Node            -
```

A Job with no GPU resource reports `-` in both GPU columns. Falcon never uses
zero current allocation as proof that a Job was CPU-only.

CPU/RAM requested values also come from the Job template. Utilization
percentages appear only when observed usage and a meaningful current
allocation denominator are both available.

`falcon metrics` is allocation-scoped: GPU and VRAM samples come only from
currently allocated devices, while CPU and memory percentages divide observed
usage by the requests of currently active Pods. Completed and queued Jobs
therefore report unavailable utilization rather than percentages against
historical requests.

Node “headroom” is allocatable minus active Pod requests. It is scheduler
request accounting, not real-time utilization. The Resources TUI Nodes pane
displays CPU, memory, and GPU as `free/allocatable`; “free” is scheduler
headroom, not measured compute or memory activity. The plain CLI summary keeps
the same explicit `free` headroom wording.

Cluster totals include Ready, schedulable Nodes. Falcon obtains the same facts
from kube-state-metrics when ordinary users cannot list Nodes: allocatable
resources, schedulability/taints, active Pod phases, and container requests.
Only Pending/Running Pod requests consume current headroom; succeeded and
failed Pods remain historical attempts, not current allocations.

Node request totals include cluster infrastructure because those requests
reduce scheduler headroom. Infrastructure Pods from Kubernetes and common
operator/monitoring namespaces are hidden from resource-consumer lists and Pod
counts so the dashboard stays focused on user workloads. This includes KEDA
and the GPU evictor.

GPU request headroom is green while at least two devices remain, yellow when
only one remains, and red when all allocatable devices are requested. Thus
15/20 is green, 1/4 is yellow, and 0/4 is red. CPU and memory headroom is green
above 70% remaining, yellow above 20% through 70%, and red at 20% remaining or
below. These are the availability equivalents of the 30% and 80% request
pressure thresholds.

When GPU Feature Discovery exports `nvidia.com/gpu.memory`, Falcon reports it
as per-device VRAM. Missing label data remains unavailable rather than being
guessed from the model name.

Kubernetes init containers use max semantics while ordinary containers use
sum semantics; Pod overhead is included. GPU extended-resource limits are
treated conservatively when malformed fixture/API data omits an equal request.

Retry fields mean:

- **Container restarts**: sum of `restartCount` across init, regular, and
  ephemeral container statuses in every Job Pod.
- **Pod attempts**: number of Pods created for the Job.
- **Succeeded/failed attempts**: terminal Pod phase counts.
- **Active Pod**: newest nonterminal attempt, if any.
- **Backoff limit**: the Job controller configuration.
