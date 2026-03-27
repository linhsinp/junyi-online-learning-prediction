# junyi-predictor Helm Chart

This chart deploys Junyi predictor batch workloads to Kubernetes without mixing runtime rollout concerns into `infra/terraform/`.

## What Helm Owns

- Kubernetes workload definitions for the repo runtime
- image, command, schedule, and runtime environment configuration
- service account wiring for cluster execution

## What Terraform Owns

- cloud resources such as GCS buckets, IAM bindings, and other shared infrastructure
- provider-level provisioning outside the cluster

## Supported Workloads

- `Job`: one-off runs such as `full_pipeline` or `preprocess_from_database`
- `CronJob`: scheduled runs such as `train_from_gcs`

## Example Commands

Render the default full-pipeline job:

```bash
helm template junyi ./infra/helm/junyi-predictor \
  -f ./infra/helm/junyi-predictor/values-full-pipeline.yaml
```

Render the scheduled GCS training workload:

```bash
helm template junyi ./infra/helm/junyi-predictor \
  -f ./infra/helm/junyi-predictor/values-train-from-gcs.yaml
```

Install the chart:

```bash
helm upgrade --install junyi ./infra/helm/junyi-predictor \
  -f ./infra/helm/junyi-predictor/values-full-pipeline.yaml
```

Set `secretEnv.*` values to reference existing Kubernetes secrets instead of hard-coding credentials in the chart.
