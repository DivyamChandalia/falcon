---
name: falcon
description: Submit and inspect Kubernetes Jobs and cluster capacity with Falcon. Use when an agent needs to launch GPU or CPU work, diagnose Job state and retries, read bounded logs, or inspect nodes and resource consumers.
---

# Falcon

Call `falcon` directly. Never wrap it in an interactive shell or `conda activate`.

## Core commands

- `falcon submit --gpu h100 --gpus 2 --name experiment -- python train.py`
- `falcon submit --cpu 8 --memory 32Gi -- python preprocess.py`
- `falcon jobs --output json --limit 50`
- `falcon get JOB --output json`
- `falcon logs JOB --tail 100`
- `falcon resources --output json`
- `falcon resources --node NODE --output json`
- `falcon delete JOB`

Prefer JSON for inspection, bound every list/log query, and treat `null` as unavailable rather than zero. Requested resources and current allocation are different facts.

Read [reference.md](reference.md) only for filters, environment selection, field meanings, or exit codes.
