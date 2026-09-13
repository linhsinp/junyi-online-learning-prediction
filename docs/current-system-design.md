# Current System Design

The current system is a Flyte 2 training workflow in `src/junyi_predictor/workflows/`, backed by date-partitioned Parquet event history, PostgreSQL dimensions, and a local or GCS artifact store. Local development uses kind and Helm PostgreSQL; the planned cloud demonstration will use Flyte OSS on ephemeral GKE.

The authoritative design, delivery stages, local runbook, and cloud boundary are documented in [the implementation plan](implementation-plan.md).

Training can also run independently through `train_from_features`, consuming
an existing v1 feature publication with a fresh experiment ID. It requires only
the artifact store and registers without promotion. Both workflow entrypoints
share chronological train-validation-test evaluation (70/15/15 by default).
The combined workflow retains its schedule and promotion behavior. See
[the standalone training runbook](training-from-features.md) for this first
increment of stage decoupling; database-backed feature publication is follow-up
work in [issue #7](https://github.com/linhsinp/junyi-online-learning-prediction/issues/7).
