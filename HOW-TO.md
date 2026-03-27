# HOW-TO

## What This System Does

This repository trains and evaluates student-performance prediction models for the Junyi learning dataset. The runtime is organized into three stage modules:

- `junyi_predictor.pipeline.preprocessing`
- `junyi_predictor.pipeline.feature_engineering`
- `junyi_predictor.pipeline.training`

Flyte orchestrates those stages through workflows in `flyte/`.

## Setup

1. Install dependencies.

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

- `flyte/full_pipeline_wf.py`: end-to-end database-backed workflow.
- `flyte/workflows/preprocess.py`: preprocessing-only workflow.
- `flyte/workflows/train_model.py`: training workflow that loads feature artifacts from GCS.
- `data/create_db.py`: utility for creating and loading PostgreSQL tables from raw CSV files.

## Outputs

- Intermediate local artifacts: `data/output/`, `data/experiment/`, `data/feature_store/`
- Model artifacts: `model/`
- Architecture reference: `docs/current-system-design.md`

## Troubleshooting

- If Flyte cannot connect to Postgres, fix `DATABASE_URL` first.
- If `flyte-train-local` fails, verify GCS credentials and bucket contents.
- If imports fail, run commands from the repository root.
- `pyflyte run` expects workflow inputs as `--start_date`, `--end_date`, and `--num_samples` rather than kebab-case flags.
