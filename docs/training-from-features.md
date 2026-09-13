# Train from existing features

This is the first delivery of [issue #7](https://github.com/linhsinp/junyi-online-learning-prediction/issues/7).
An independent experiment reuses a retained feature snapshot without connecting
to PostgreSQL or invoking preprocessing/feature engineering. The combined
workflow and its weekly schedule remain available.

## Run an experiment

Find the producer run's `feature_snapshot.json` under the configured artifact
root. Supply its **store-relative key**, not a filesystem path or GCS URL:

```sh
make flyte-train-from-features-local \
  FEATURE_SNAPSHOT_KEY='runs/<source-run-id>/feature_snapshot.json'
```

Replace `<source-run-id>` with an existing producer run ID whose feature
generation completed; its model training need not have succeeded. The quotes
prevent the shell from interpreting the placeholder's angle brackets as file
redirection. The command generates a new training-run ID. To name the experiment and select
an optional interval:

```sh
make flyte-train-from-features-local \
  FEATURE_SNAPSHOT_KEY='runs/<source-run-id>/feature_snapshot.json' \
  TRAINING_RUN_ID=experiment-01 \
  FEATURE_START_DATE=2019-06-01T00:00:00 \
  FEATURE_END_DATE=2019-06-10T00:00:00 \
  TRAIN_FRACTION=0.70 VALIDATION_FRACTION=0.15
```

`FEATURE_START_DATE` and `FEATURE_END_DATE` are deliberately separate from the
combined workflow's sample `START_DATE`/`END_DATE` defaults. Omit either bound
to leave that side unbounded. The Make target uses local artifacts, honors
`ARTIFACT_ROOT` (default `artifacts/runs`), and loads `.env` like the existing
targets. `DATABASE_URL` is not required or used by standalone training.

The Flyte entrypoint also works with GCS using `ARTIFACT_BACKEND=gcs`,
`GCS_BUCKET`, and `ARTIFACT_ROOT` as the bucket prefix. For example, from an
environment with those settings and cloud credentials:

```sh
PYTHONPATH=src uv run flyte run --local \
  src/junyi_predictor/workflows/training.py train_from_features \
  --feature_snapshot_key 'runs/<source-run-id>/feature_snapshot.json'
```

Its optional arguments are `--training_run_id`, `--start_date`, `--end_date`,
`--train_fraction`, and `--validation_fraction`. Run IDs must begin with an
alphanumeric character and contain only alphanumerics, underscores, hyphens,
or periods. An already registered run ID is rejected before input loading;
use a fresh ID for a new experiment.

## Dataset selection and evaluation

“All data” means all rows in the chosen feature snapshot are selected **before**
the chronological train-validation-test split. This transitional entrypoint
does not combine snapshots or query all historical database features.

Date filters run first and apply the same positional mask to the event rows
and both feature matrices. Bounds are UTC, start-inclusive and end-exclusive;
naive bounds are interpreted as UTC. Source timestamps must already be ordered.

The defaults are 70% training, 15% validation, and the remaining 15% test.
Both supplied fractions must be positive and their sum must be below one.
Nominal cut positions are the floors of `N * train_fraction` and
`N * (train_fraction + validation_fraction)`. A cut inside an equal-timestamp
group moves forward to the end of that group, so actual proportions can differ.
Selections with an empty partition or only one training target class fail
explicitly; the implementation does not silently resample or drop invalid rows.

The scaler and four existing model candidates fit only on the training
partition. Validation accuracy selects the winner, with ties resolved in the
existing candidate order. Only that winner is evaluated on held-out test data;
it is not refitted on validation or test rows.

The combined workflow now uses this same evaluation and accepts the same
fraction arguments (`TRAIN_FRACTION`/`VALIDATION_FRACTION` in its Make targets).
Its previous 80/20 helper APIs remain available to existing Python callers,
but workflow execution uses three partitions. New candidate metrics contain
`train_score` and `validation_score`; only the selected model has `test_score`.
Older scores used for candidate selection are not equivalent to these new
held-out test scores.

## Outputs and reuse guarantees

Each experiment writes `models/<new-training-run-id>/` within the artifact root:

- `model.joblib`, `scaler.joblib`, and candidate `metrics.json`.
- `training_metadata.json`: producer identity, original manifest, SHA-256
  fingerprints, requested dates, actual partition boundaries/counts, feature
  ordering and matrix shapes/dtypes, estimator configurations/seeds, library
  versions, and evaluation-protocol version.
- `manifest.json`: model registration, including `training_metadata_uri`.

Standalone execution **registers only** and leaves `models/approved.json`
unchanged. The combined workflow still promotes its selected candidate.
Existing registration manifests without `training_metadata_uri` remain readable.

Existing v1 publications use the producer's canonical files under
`runs/<source-run-id>/features/`: `log.parquet`, `concept_proficiency.npy`, and
`level4_proficiency.npy`. The loader uses those keys in the configured store;
manifest URI fields are retained as provenance. Moving the complete retained
publication to another store root therefore does not require rewriting its
manifest. Temporary private copies are fingerprinted and loaded, then removed
on success or failure; source files are not rewritten.

Retain the original publication for reproducibility. Fingerprints identify the
bytes consumed but do not enforce source immutability, guarantee coherent reads
during concurrent source rewrites, or recover deleted data. Legacy snapshots
lack semantic concept-to-column mappings; metadata records the existing matrix
block order and positional representation without inventing labels.

Database feature publication, incremental ingestion, independent schedules,
promotion policy, and concurrent publication/retry recovery remain tracked by
issue #7. This change requires no database migration or reset.

Hyperparameters and predictor selection still live in Python. A proposed
[experiment YAML and MLflow tracking follow-up](experiment-configuration.md)
will provide scientist-facing configuration and experiment comparison; these
capabilities are not part of this first PR.
