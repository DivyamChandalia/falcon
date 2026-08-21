# Falcon resource service administration

Falcon clients use the read-only service at
`http://node1.yoda.hyperverge.org:30081`. It is the sole poller of the local
kube-state-metrics endpoint and the sole owner of the new 24-hour GPU
allocation history. Existing per-user SQLite databases are neither migrated
nor deleted.

## Installation on node1

Install the Python package and create the locked-down service identity before
copying the unit files:

```console
python3 -m pip install /path/to/falcon-k8s
sudo useradd --system --home /var/lib/falcon-resource-service \
  --shell /usr/sbin/nologin falcon-resource
sudo install -m 0644 deploy/falcon-resource-service.{socket,service} /etc/systemd/system/
sudo systemd-analyze verify /etc/systemd/system/falcon-resource-service.{socket,service}
sudo systemctl daemon-reload
sudo systemctl enable --now falcon-resource-service.socket falcon-resource-service.service
```

The socket is fixed to private address `192.168.1.117:30081`, uses
`Accept=no`, and is passed to the single long-running service. The process also
holds an exclusive lock in `/var/lib/falcon-resource-service`, so a manually
started second publisher cannot open the history database. Do not expose port
30081 outside the trusted private cluster network.

`falcon setup` only writes user configuration and shell completion. It does
not request sudo or install this system service.

## Verification and rollout

Install and verify the server before rolling out clients whose new default is
`cluster.resource_service_url`. From node1 and from a Coder Pod, run:

```console
curl --fail http://node1.yoda.hyperverge.org:30081/healthz
curl --fail http://node1.yoda.hyperverge.org:30081/v1/snapshot
curl -N http://node1.yoda.hyperverge.org:30081/v1/stream
```

The stream sends one initial snapshot and then remains completely silent until
resource state changes. `systemctl status falcon-resource-service` should show
one service, and `/healthz` exposes the revision, subscriber count, history
size, freshness, and upstream-request count. With several clients connected,
the upstream count must still increase only once per five-second interval.

Upgraded clients safely stop their own positively identified legacy history
process. Administrators should audit collectors owned by users who have not
yet upgraded (for example, processes whose command line contains
`-m falcon.resources_history`) and ask those users to upgrade before stopping
them. Do not remove their personal SQLite files.

For deployments without this server, explicitly configure:

```yaml
cluster:
  resource_service_url: null
```

That setting retains the legacy direct kube-state-metrics/Kubernetes and local
history behavior.
