# Infrastructure Design

```mermaid
flowchart TD
    subgraph Infra[Infrastructure Under infra/]
        TF[infra/terraform<br/>cloud provisioning]
        DOCKER[infra/docker<br/>container build definitions]
        HELM[infra/helm/junyi-predictor<br/>Kubernetes packaging]
    end

    subgraph Provisioned[Provisioned Resources]
        BUCKET[GCS bucket]
        IAM[Service account and IAM]
        CLUSTER[Kubernetes cluster]
    end

    subgraph Runtime[Deployed Workloads]
        JOB[Job<br/>full_pipeline or preprocess_from_database]
        CRON[CronJob<br/>train_from_gcs]
    end

    TF --> BUCKET
    TF --> IAM
    TF --> CLUSTER

    DOCKER --> HELM
    HELM --> JOB
    HELM --> CRON

    BUCKET --> CRON
    IAM --> JOB
    IAM --> CRON
    CLUSTER --> JOB
    CLUSTER --> CRON
```

## Notes

- `infra/terraform/` owns shared cloud resources and provider-side provisioning.
- `infra/docker/` owns buildable container images for runtime workloads.
- `infra/helm/junyi-predictor/` owns Kubernetes workload definitions and runtime configuration.
- Helm and Terraform are intentionally separated so cluster rollout logic does not get mixed into cloud provisioning.
