# HOW-TO

## What This System Does

This repository trains and evaluates student-performance prediction models for the Junyi learning dataset. The runtime is organized into three stage modules:

- `junyi_predictor.pipeline.preprocessing`
- `junyi_predictor.pipeline.feature_engineering`
- `junyi_predictor.pipeline.training`

Flyte 2 orchestrates durable training through tasks in `src/junyi_predictor/workflows/training.py`.

## Setup

This repository is expected to run only from the `uv`-managed virtual environment.

1. Sync the environment.

```bash
uv sync --all-groups
```

2. For cloud-free integration work, install PostgreSQL into kind and use its NodePort connection string.

## Run Tests

Use tests as the first validation step before running workflows.

```bash
make test
make lint
```

## Run the local end-to-end workflow

Create kind and install PostgreSQL through Helm:

```bash
make kind-create
make postgres-local
```

Download the Kaggle dataset, convert the event history into partitioned Parquet,
then seed only the small relational dimension tables in PostgreSQL:

```bash
make download-data
make materialize-parquet
make reset-local # required before reseeding; drops local source tables
make seed-local
```

Run the durable Flyte workflow locally. It uses the filesystem artifact store under `artifacts/runs/`:

```bash
DATABASE_URL=postgresql://junyi:junyi-local-only@localhost:30001/junyi \
make flyte-training-local START_DATE=2019-06-01T00:00:00 END_DATE=2019-06-10T00:00:00
```

## Remote GKE demonstration

Use Terraform and Helm for different responsibilities:

- `infra/terraform/demo/` provisions an ephemeral GKE, Cloud SQL, GCS, Artifact Registry, and Workload Identity environment.
- `infra/helm/flyte/` configures the Flyte OSS control plane.
- `infra/docker/` contains the single runtime image definition.

Bootstrap a Terraform state bucket once, then initialize the destroyable GKE environment:

```bash
terraform -chdir=infra/terraform/bootstrap init
terraform -chdir=infra/terraform/bootstrap apply
terraform -chdir=infra/terraform/demo init \
  -backend-config="bucket=<state-bucket>" \
  -backend-config="prefix=junyi/demo"
```

Build and push the runtime image, register the workflow with `flyte deploy`, run one remote execution, then destroy the environment after the demonstration:

```bash
terraform -chdir=infra/terraform/demo destroy
```

Remote tasks use Workload Identity and `ARTIFACT_BACKEND=gcs`; do not mount service-account JSON keys.

## Main Entry Points

- `src/junyi_predictor/workflows/training.py`: composed Flyte training workflow.
- `src/junyi_predictor/pipeline/`: plain preprocessing, feature, and training stages.
- `src/junyi_predictor/bootstrap/`: Kaggle download and PostgreSQL seeding.
- `infra/docker/`: the single runtime image.
- `infra/helm/local-postgres/`: local kind PostgreSQL chart.
- `infra/helm/flyte/`: Flyte OSS values overlay.
- `infra/terraform/`: bootstrap and cloud demonstration provisioning.

## Outputs

- Curated training events: `artifacts/data/curated/log_problem/year=*/month=*/`
- Run-scoped artifacts: `artifacts/runs/runs/<run-id>/`
- Registered models: `artifacts/runs/models/<model-version>/`
- Architecture reference: `docs/current-system-design.md`

## Troubleshooting

- If Flyte cannot connect to Postgres, fix `DATABASE_URL` first.
- If `flyte-training-local` fails, verify the kind PostgreSQL release and its NodePort connection string.
- Source code lives under `src/`; use the provided Make targets for operational
  commands. If calling a module directly, prefix it with `PYTHONPATH=src`.
- This repo targets the Flyte 2 `flyte` CLI, not `pyflyte`, and local runs use `flyte run --local ...`.
- Do not use `pyenv`, `python -m venv`, or ad hoc `pip install`; use `uv sync` and `uv run` only.
