# Cloud configuration ownership and migration plan

## Target ownership model

Configuration is owned by the layer it governs, not by the workload that consumes it.

| Concern | Owner | Examples |
| --- | --- | --- |
| Cloud foundation and authorization | Terraform | VPC, GKE, Cloud SQL, GCS, Artifact Registry, IAM, Workload Identity |
| Flyte platform | Helm: `infra/helm/flyte` | Pinned chart, control-plane configuration, private service, platform resources |
| Junyi project deployment policy | Helm: new `infra/helm/junyi-cloud` | Task service account, ConfigMap, Secret reference, quota, seeder Job |
| Task execution contract | Python workflows | Task graph, runtime image, per-task resources, PodTemplate references |

Use this contributor rule:

1. Create, network, store, or authorize a cloud resource: Terraform.
2. Deploy or configure software in Kubernetes: Helm.
3. Define how a Junyi task executes: workflow code.

A task resource request describes one pod's needs. A ResourceQuota constrains the Junyi namespace as a whole. Requests belong to workflow code; the quota belongs to the Junyi Helm release.

## Current state

The cloud MVP Terraform currently creates Junyi Kubernetes objects: the task service account, task ConfigMap, database Secret, ResourceQuota, seeder Job, and Flyte Helm release. This is transitional and does not match the target model.

## Implementation plan

### 1. Add a Junyi cloud chart

Create `infra/helm/junyi-cloud/`. Its values and templates must render the `junyi-flyte-task` service account with a supplied Workload Identity annotation, task ConfigMap, database Secret reference, `junyi-demo-limits` ResourceQuota, and a disabled-by-default seeder Job. Enable the Job only when supplied an immutable runtime image. Keep service-account, ConfigMap, and Secret names aligned with the Python PodTemplate constants.

Add rendering tests for the identity annotation, quota, ConfigMap, and both seeder-Job paths.

### 2. Restrict Terraform to cloud and authorization

Remove Kubernetes/Helm providers, `helm_release`, and all Kubernetes resources from `infra/terraform/demo`. Retain GCP resources, Google service accounts, bucket IAM, and Workload Identity IAM bindings, which refer to documented Helm-created service-account names.

Expose non-secret outputs for the image repository, buckets, Cloud SQL private address, Google service-account emails, cluster name, and region. Terraform may receive the database password as sensitive input, but must not render it into a Kubernetes manifest.

### 3. Make Helm the cluster deployment interface

After Terraform apply, install `junyi-cloud` with cloud outputs, task Google service-account email, and an ignored sensitive values file for the database URL. Install `flyte-binary` with its checked-in overlay, cloud outputs, control-plane Google service-account email, and an ignored sensitive values file for the Flyte database password.

Build and push the image before a second `junyi-cloud` upgrade enables the seeder Job. Wait for the Job, then register and run Flyte as today. Keep the control plane private and use port-forward for the demo.

### 4. Update operations and validate

Replace Terraform Helm/seeder commands in the cloud-MVP runbook with the two Helm release commands and generated, ignored values files. Never commit database URLs or passwords.

Require Terraform validation, Helm rendering for both charts, Python tests, Ruff, and a manually gated one-day remote run. Verify rendered identity annotations, task requests, quota enforcement, and immediate teardown.

## Rollout order

First land the chart and render tests. Next, remove Terraform's equivalent Kubernetes resources and update the runbook in the same change. Do not let Helm and Terraform manage copies of the same Kubernetes object concurrently.
