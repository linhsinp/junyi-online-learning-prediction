# HOW-TO

## What This System Does

This repository trains and evaluates student-performance prediction models for the Junyi learning dataset. The runtime is organized into three stage modules:

- `junyi_predictor.pipeline.preprocessing`
- `junyi_predictor.pipeline.feature_engineering`
- `junyi_predictor.pipeline.training`

Flyte 2 orchestrates those stages through tasks in `orchestration/flyte_app.py`, using the `flyte` CLI for local execution.

## Setup

This repository is expected to run only from the `uv`-managed virtual environment.

1. Sync the environment.

```bash
uv sync --all-groups
```

2. Make sure `.env` contains a working `DATABASE_URL`.
3. If you want to run the GCS-backed training workflow, ensure Google Cloud credentials are available locally.

## Run Tests

Use tests as the first validation step before running workflows.

```bash
make test
make lint
```

## Deploy to Kubernetes

Use Terraform and Helm for different responsibilities:

- `infra/terraform/` provisions cloud resources such as the GCS bucket and IAM bindings.
- `infra/helm/junyi-predictor/` defines Kubernetes runtime workloads and cluster-level configuration.
- `infra/docker/` contains the Dockerfiles used to build runtime images.

Render the default one-off training job:

```bash
helm template junyi ./infra/helm/junyi-predictor \
  -f ./infra/helm/junyi-predictor/values-full-pipeline.yaml
```

Install a scheduled GCS-backed training workload:

```bash
helm upgrade --install junyi ./infra/helm/junyi-predictor \
  -f ./infra/helm/junyi-predictor/values-train-from-gcs.yaml
```

Point `secretEnv` entries in the values files at existing Kubernetes secrets for `DATABASE_URL` or cloud credentials.

## Run Flyte Locally

Run the end-to-end workflow:

```bash
make flyte-local START_DATE=2019-06-01T00:00:00 END_DATE=2019-06-10T00:00:00 NUM_SAMPLES=1000
```

Run only preprocessing:

```bash
make flyte-preprocess-local START_DATE=2019-06-01T00:00:00 END_DATE=2019-06-10T00:00:00
```

Run the GCS-backed training workflow:

```bash
make flyte-train-local
```

## Main Entry Points

- `orchestration/flyte_app.py`: Flyte 2 task entrypoints for preprocessing, full pipeline execution, and GCS-backed training.
- `junyi_predictor/pipeline/preprocessing.py`: preprocessing stage contract and transformations.
- `junyi_predictor/pipeline/feature_engineering.py`: feature engineering stage contract and transformations.
- `junyi_predictor/pipeline/training.py`: training split and model execution helpers.
- `junyi_predictor/bootstrap/database.py`: utility for creating and loading PostgreSQL tables from raw artifact files.
- `junyi_predictor/bootstrap/kaggle.py`: utility for downloading the raw Kaggle dataset into local artifacts.
- `infra/docker/`: container build definitions for local and cluster execution.
- `infra/helm/junyi-predictor/`: Kubernetes packaging for one-off and scheduled runtime workloads.
- `infra/terraform/`: cloud infrastructure provisioning.

## Outputs

- Intermediate local artifacts: `artifacts/data/output/`, `artifacts/data/experiment/`, `artifacts/data/feature_store/`
- Model artifacts: `artifacts/model/`
- Architecture reference: `docs/current-system-design.md`

## Troubleshooting

- If Flyte cannot connect to Postgres, fix `DATABASE_URL` first.
- If `flyte-train-local` fails, verify GCS credentials and bucket contents.
- If imports fail, run commands from the repository root.
- This repo targets the Flyte 2 `flyte` CLI, not `pyflyte`, and local runs use `flyte run --local ...`.
- Do not use `pyenv`, `python -m venv`, or ad hoc `pip install`; use `uv sync` and `uv run` only.
