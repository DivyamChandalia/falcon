# Falcon agent reference

## Inspection

Use one bounded command per question:

```text
falcon jobs --status running --output json --limit 50
falcon jobs --status failed --output json --limit 50
falcon get JOB --output json
falcon logs JOB --tail 100
falcon resources --gpu h100 --output json
falcon resources --node NODE --output json
```

`get` is the authoritative Job view for requested resources, current allocation, attempts, container restarts, active pod, and recent events. In generic GPU fields, `null`/`-` means no request or no current allocation; it does not prove a completed Job was CPU-only.

## Submission

Pass the workload after `--` so its arguments remain separate:

```text
falcon submit --gpu h100 --gpus 2 --name experiment -- python train.py
falcon submit --cpu 8 --memory 32Gi -- python preprocess.py
```

Use `falcon submit --help` to select or override a Python environment. Never synthesize `zsh -lic`, `source`, or nested shell quoting. Verify a manifest with dry-run before an uncertain or high-cost submission.

## Output and errors

JSON has no ANSI or surrounding log text. Read errors from stderr.

- `0`: success
- `2`: invalid arguments or configuration
- `3`: Kubernetes or telemetry unavailable
- `4`: requested object not found
- `130`: interrupted

Do not retry invalid input. Retry transient availability failures with a bounded delay. Before deleting a Job, confirm its exact identity from `get`.
