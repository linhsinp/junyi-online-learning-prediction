# Local Flyte backend compatibility preflight

This disposable preflight proves the boundary that `make flyte-training-local`
does not cover: Flyte SDK 2.0.12 submits a task to the exact backend image
packaged by the pinned Flyte chart (`v2.0.20`), which schedules the fixed
linux/amd64 Junyi runtime image in Kubernetes.

It uses the existing `kind-junyi` cluster, but creates an independent
`flyte-preflight` namespace. It never uses or modifies the `junyi-local`
PostgreSQL database. The preflight namespace contains its own PostgreSQL
metadata database and ephemeral MinIO bucket; both are destroyed at teardown.

## Run

Make sure the current Kubernetes context is the existing local kind cluster:

```sh
kubectl config use-context kind-junyi
make flyte-backend-preflight-image
make flyte-backend-preflight-up
make flyte-backend-preflight-run
make flyte-backend-preflight-status
```

The final run must complete successfully and report:

```text
flyte_sdk_version: 2.0.12
task_service_account: junyi-flyte-task
task_config_injected: true
task_secret_injected: true
```

`flyte-backend-preflight-image` builds the same `linux/amd64` runtime image
used for the cloud runbook and loads it into the existing kind cluster. The
probe uses the production task pod template, so it verifies the Helm-managed
service account, ConfigMap, and Secret are available to a Flyte-scheduled pod.
It reports only booleans for configuration injection and never prints the
database URL or a service-account token.

The run target creates the isolated `flyte-preflight` Flyte project on first
use and deploys into its `development` domain. No global Flyte CLI config file
is required. During this command only, MinIO is port-forwarded at port 9000 so
the host CLI and kind task pods share one S3-compatible endpoint.

## Teardown

```sh
make flyte-backend-preflight-down
```

This removes only the `flyte-preflight` namespace and its Helm releases. It
does not delete the kind cluster or the `junyi-local` namespace.
