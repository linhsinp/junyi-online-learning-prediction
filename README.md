Junyi Online Learning Performance Prediction
=============================================

*This repo is an on-going project and is actively evolving!* 

This project aims to build a prototype tool that is applicable to a Taiwanese online learning platform. A best-performing ML model among other candidates makes batch inference on incoming online learning platform data. It predicts students' online learning performance based on their problem-solving history, which sheds light on ways to help students improve their academic performance.

It demonstrates how to orchestrate data and model execution with Flyte 2 while keeping pipeline logic modular and testable in plain Python. Local runs use Flyte local mode against kind PostgreSQL and a filesystem-backed Parquet data lake. Flyte OSS on ephemeral GKE is the planned remote demonstration; its Terraform and Helm configuration is present but the remote deployment is follow-up work.

Local development is expected to use the `uv`-managed virtual environment only.

The end-to-end ML pipeline includes:

1. Data ingestion - raw CSV events materialized as date-partitioned Parquet
2. Data preprocessing - a run-scoped, durable preprocessing snapshot
3. Feature engineering - a run-scoped feature snapshot and relational run output
4. Model training, evaluation, and registration
5. Batch inference - follow-up work
6. Data and model monitoring - follow-up work

Infrastructure is separated by concern under `infra/`:

- `infra/terraform/`: provisions shared cloud resources such as buckets and IAM
- `infra/helm/local-postgres/`: development-only PostgreSQL for a kind cluster
- `infra/helm/flyte/`: Flyte OSS deployment overlay for GKE
- `infra/docker/`: builds the container images used by local and cluster workloads

See [the implementation plan](docs/implementation-plan.md) for the local kind,
Flyte, and ephemeral-GKE delivery path.

For the local kind workflow, use `make kind-create`, `make postgres-local`,
`make download-data`, `make materialize-parquet`, `make seed-local`, and
`make flyte-training-local` from the repository root. The full event history is
stored as date-partitioned Parquet under `artifacts/data/curated/`; PostgreSQL
contains only relational dimensions and run outputs. The workflow executes
preprocess, feature, and training as separate Flyte local actions. Use
`make flyte-training-local-tui` to observe a newly launched local run
interactively. The application source is under `src/`; the Make targets set the
required source path for module-based commands.

Use `make reset-local` before reseeding an existing local database. It drops
the local source tables and their contents before they are recreated by
`make seed-local`.


Open source dataset on Kaggle: [Junyi Academy Online Learning Activity Dataset](https://www.kaggle.com/datasets/junyiacademy/learning-activity-public-dataset-by-junyi-academy/)
