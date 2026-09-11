# Current System Design

The current system is a Flyte 2 training workflow in `src/junyi_predictor/workflows/`, backed by date-partitioned Parquet event history, PostgreSQL dimensions, and a local or GCS artifact store. Local development uses kind and Helm PostgreSQL; the cloud demonstration uses Flyte OSS on ephemeral GKE.

The authoritative design, delivery stages, local runbook, and cloud boundary are documented in [the implementation plan](/Users/hsin-pei/Desktop/github_repo/junyi-online-learning-prediction/docs/implementation-plan.md).
