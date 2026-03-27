# Local Docker and Kubernetes Testing

## Build Docker Images Locally

Run builds from the repository root so `COPY . /app` includes the full project context.

```bash
docker build -f infra/docker/Dockerfile -t junyi-predictor:local .
docker build -f infra/docker/train/Dockerfile -t junyi-predictor-train:local .
docker build -f infra/docker/batch_infer/Dockerfile -t junyi-predictor-batch:local .
```

Smoke-test the main image:

```bash
docker run --rm junyi-predictor:local uv run flyte --help
```

## Test on a Local Kubernetes Cluster

`kind` is the simplest local cluster option.

1. Create the cluster.

```bash
kind create cluster --name junyi
```

2. Load the local image into the cluster.

```bash
kind load docker-image junyi-predictor:local --name junyi
```

3. Create the secrets required by the workload.

Example for database-backed jobs:

```bash
kubectl create secret generic junyi-predictor-db \
  --from-literal=DATABASE_URL='postgresql://user:pass@host:5432/dbname'
```

4. Install the Helm chart with the local image.

```bash
helm upgrade --install junyi ./infra/helm/junyi-predictor \
  -f ./infra/helm/junyi-predictor/values-full-pipeline.yaml \
  --set image.repository=junyi-predictor \
  --set image.tag=local \
  --set image.pullPolicy=IfNotPresent
```

5. Inspect the created job.

```bash
kubectl get jobs
kubectl describe job junyi-junyi-predictor
kubectl logs job/junyi-junyi-predictor
```

## Scheduled Workload Check

To test the scheduled GCS-backed workload:

```bash
helm upgrade --install junyi ./infra/helm/junyi-predictor \
  -f ./infra/helm/junyi-predictor/values-train-from-gcs.yaml \
  --set image.repository=junyi-predictor \
  --set image.tag=local \
  --set image.pullPolicy=IfNotPresent
kubectl get cronjobs
```

## Notes

- The chart runs `flyte run --local orchestration/flyte_app.py ...` inside the container.
- Database-backed jobs require `DATABASE_URL`.
- GCS-backed jobs require cloud credentials and the expected bucket contents.
- Keep the Docker build context as `.` from the repository root.
