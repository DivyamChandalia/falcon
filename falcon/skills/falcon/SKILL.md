---
name: falcon
description: Launch, observe, log, and kill Kubernetes Jobs with Falcon. Use for GPU or CPU Jobs requested by the user.
---

# Falcon

Call `falcon` directly. Use the same commands a human uses; never wrap Falcon in an interactive shell or `conda activate`.

## Commands

- `falcon h100x2 --name experiment -- python train.py`
- `falcon -c 8 -m 32Gi -- python preprocess.py`
- `falcon logs JOB --no-follow --tail 100`
- `falcon metrics JOB --interval 60`
- `falcon kill JOB`

## Behavior

- Launch the Job, report its name, and return. Do not add `-f`, wait, poll, fetch logs, or observe utilization unless the user explicitly asks to follow or verify it.
- If the user asks to observe the Job or ensure utilization is sufficient, use `metrics`; its GPU, VRAM, CPU, and memory percentages are based on current allocation.
- Eviction policy uses average GPU utilization: H100 must stay at or above 90%; A6000 and 2080Ti must stay at or above 30%.
- Falcon automatically leaves a 1 GiB memory safety buffer. Use `-m` only when the user explicitly requests a memory value or the workload demonstrably needs it.
- For status, events, manifests, scheduling details, or anything outside launch/log/metrics/kill, use one targeted `kubectl` command.
