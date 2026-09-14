# Proposed DVC and MLflow integration

**Status:** Proposed; deferred until after the cloud MVP and reconciliation of
[issue #7](https://github.com/linhsinp/junyi-online-learning-prediction/issues/7)'s
storage requirements. Neither integration is implemented by this document.

The [canonical implementation plan](implementation-plan.md) remains the delivery
reference. This document preserves a two-stage proposal building on
[PR #8](https://github.com/linhsinp/junyi-online-learning-prediction/pull/8)'s
[independent training entrypoint](training-from-features.md).

## Delivery priority and prerequisites

The recommended order is to close the correctness and execution gaps needed for
one credible remote run, demonstrate the cloud MVP, and then resume broader
stage decoupling and evaluate these integrations. Completing issue #7 as written
is not recommended as a prerequisite for that demonstration.

The canonical first remote milestone is bounded: train and register one approved
model on ephemeral GKE, inspect the evidence, and destroy the environment. Its
existing architecture uses Parquet history, PostgreSQL dimensions, and durable
artifact handoffs. Issue #7 additionally requires daily ingestion, complete
PostgreSQL feature storage, immutable revisions, incremental feature state,
independent schedules, concurrency recovery, and performance benchmarks.

Issue #7 still explicitly requires PostgreSQL as the source of complete feature
data. The architectural assessment motivating this proposal favors retained
snapshots for batch feature engineering and training. Reconcile that choice
before implementing storage-specific follow-ups. This document does not revise
the issue, mark its acceptance criteria complete, or replace its current scope.

| Priority | Recommended work | Evidence or outcome |
| --- | --- | --- |
| Now | Preserve this deferred proposal | A reviewable design without adding runtime components |
| Next | Close local correctness and cloud-readiness gaps | A representative run with measured resources and verified outputs |
| Then | Execute the bounded cloud MVP | Remote task execution, data/artifact handoffs, approval, and teardown evidence |
| Afterward | Reconcile and resume issue #7; revisit these integrations | A storage decision informed by execution evidence |

The readiness assessment should cover:

- A representative local run with per-stage duration and memory measurements,
  verified artifacts and lineage, chronological evaluation, and retained-input
  training. The reported long-running experiment remains unexplained; no speedup
  is assumed.
- Explicit remote access to selected event data. Preprocessing currently reads
  local Parquet paths; configuring GCS model artifacts alone does not make those
  inputs available to remote workers.
- Compatibility of the pinned Flyte SDK, control plane, chart, and runtime image,
  plus task identity, Cloud SQL connectivity, and durable artifact handoffs.
- Fixed demonstration inputs, fresh run IDs, disabled recurring schedules, and a
  documented cleanup/retry procedure. Correctness defects discovered on this path
  must be fixed; generalized incremental ingestion and concurrent publication
  recovery remain separate work.
- Recorded remote task executions, database outputs, GCS artifacts, approved
  manifest, and successful teardown.

PR #8 demonstrates core reuse with fixture data. That supports the cloud
milestone but does not establish completion of issue #7. The sequencing above is
a recommendation, not a record of completed validation or a replacement roadmap.

## Integration boundaries

Deliver the integrations in two sequential PRs. Both are local-first, with
cloud-ready configuration and documentation. Cloud provisioning and remote
demonstrations of these integrations remain follow-up work.

| Responsibility | Owner |
| --- | --- |
| Execution, scheduling, dependencies, retries, and task resources | Flyte |
| Released dataset bytes and version retrieval | DVC remote and Git metadata |
| Feature meaning, historical context, mappings, and selection/splits | Junyi contracts and manifests |
| Experiment search, comparisons, and metrics | MLflow |
| Canonical model bundles and approved-model pointer | Existing Junyi registry |

Automatic upstream publication, ingestion redesign, hyperparameter search, and
migration to MLflow Model Registry are outside these two PRs. The existing
[validated experiment YAML proposal](experiment-configuration.md) remains
separate; YAML is not implemented or made a prerequisite here. If delivered
first, its resolved configuration should be included in experiment reports.

Neither DVC nor MLflow supplies event identity, historical feature correctness,
or publication completeness automatically. Dataset references and their
retention policies must support those application contracts.

## Stage 1: DVC-backed feature releases

**Proposed PR:** `:sparkles: train from DVC-versioned feature releases`

### Publication and storage

- Initialize DVC in the existing Git repository. Track release metadata under
  `datasets/features/`; keep dataset bytes and caches out of Git.
- Add a manual `prepare-feature-release` command accepting an existing feature
  snapshot key and a new dataset ID. Copy the complete retained publication into
  an isolated release directory without modifying its source.
- Package the event Parquet, both proficiency matrices, original manifest, and a
  release manifest containing file SHA-256 hashes, row count, schema, source
  identity, and feature layout.
- Preserve legacy positional matrix semantics explicitly. Do not invent concept
  labels absent from the original publication.
- Validate the bundle before `dvc add`. Reject an existing dataset ID; corrections
  produce a new release.
- Document publication order: prepare and validate, `dvc add`, `dvc push`, then
  commit/push DVC metadata with Git. Commands must not automatically commit or
  push Git changes. Training workers only read published releases.
- Use a local DVC remote for development and tests. Document a separately
  configured GCS remote and application-default credentials. Retained release
  data must live outside the disposable demo environment.

### Training interface

- Add a frozen `DvcDatasetReference` contract containing repository location,
  full Git commit SHA, and tracked dataset path. Reject moving branch/tag
  references at the training boundary.
- Add `train_from_dvc` with that reference plus the existing optional training-run
  ID, date bounds, and split fractions. Provide a corresponding Make target.
- Retrieve the selected directory into a temporary workspace and verify its
  release manifest before fitting.
- Refactor the loader boundary so artifact-backed and DVC-backed inputs use the
  same selection, evaluation, and registration logic. Keep model outputs in the
  existing configured artifact store.
- Record the DVC reference, release ID, hashes, code revision, and existing
  selection/split metadata in the model's training metadata. Record the code
  revision separately from the dataset revision.
- Preserve `train_from_features` and the combined workflow. Standalone DVC
  training registers without promotion.
- Keep publication and registration uncached; introduce no new Flyte computation
  caching in this increment. Any later computation cache must include pinned
  dataset identity and resolved configuration in its inputs.

### Acceptance

- [ ] A clean workspace retrieves a release using only its Git metadata and DVC
  remote.
- [ ] Two real Flyte local training runs use identical released inputs while
  database and upstream entrypoints are configured to fail if called.
- [ ] A newer release does not prevent retrieval and training from the original
  commit.
- [ ] Missing objects, invalid references, hash mismatches, and malformed bundles
  fail before fitting.
- [ ] Original artifacts and the approved-model pointer remain unchanged;
  temporary downloads are cleaned up.
- [ ] Release retention is documented. Routine remote garbage collection is
  prohibited until its retained-reference policy is established.

## Stage 2: Optional MLflow experiment tracking

**Proposed PR:** `:sparkles: track standalone training experiments with MLflow`

### Integration and interfaces

- Use the MLflow Python SDK directly, retaining the current Flyte version. Defer
  adoption of the newer Flyte MLflow plugin; its current documentation does not
  establish compatibility with the repository's pinned Flyte 2.0.11 environment.
- Add optional tracking configuration: enabled flag, tracking URI, and experiment
  name. Default tracking to disabled.
- Supply a local MLflow server Make target with an explicit SQLite backend and
  local artifact directory under gitignored `artifacts/`.
- Support a remotely reachable tracking URI through the same configuration.
  Document shared-server deployment with a separate PostgreSQL metadata database
  and durable artifact storage; do not provision it in this PR.
- Add an immutable experiment-report artifact containing resolved candidate
  parameters, metrics, dataset identity, code/environment information, selection,
  and splits. Extend new registrations with an optional report URI; old manifests
  remain readable.

### Experiment organization and recovery

- Keep fitting and model registration independent of MLflow. After successful
  standalone registration, invoke a separate tracking-publication task.
- Create one MLflow parent run per Junyi training run and one child per candidate,
  with stable candidate IDs.
- Log training and validation metrics per candidate; log held-out test
  performance only for the winner.
- Attach the complete lineage/report and canonical model-manifest URI. Do not
  duplicate training datasets or automatically register models in MLflow Model
  Registry.
- Add `sync-experiment-tracking` to publish an existing report without reading
  training data or fitting models.
- Persist MLflow IDs in a separate synchronization receipt. Resume known runs on
  replay; use Junyi run/candidate tags to recover interrupted publication before
  a receipt was saved.
- Support one synchronization writer per training run. Do not claim exactly-once
  behavior under concurrent publishers.
- Tracking errors use bounded retries and produce a visible failed/pending
  tracking status. The completed model registration remains successful and can
  be synchronized later.
- Initially track successful standalone experiments only. Flyte and structured
  logs remain authoritative for training failures.

### Acceptance

- [ ] A real local MLflow server displays one parent and the expected candidate
  children with matching lineage and metrics.
- [ ] Both artifact-backed and DVC-backed standalone training can publish reports.
- [ ] Tracking disabled preserves existing behavior.
- [ ] Tracking-server failure leaves a usable model registration and replayable
  report.
- [ ] Replaying synchronization resumes the same experiment without refitting or
  duplicating candidates in the supported single-writer path.
- [ ] No loser receives a test score and no tracking operation changes approval.
- [ ] Tracking can be reconstructed from the stored report after the training
  workspace is removed.

## Validation and rollout

- Manage dependencies exclusively through `uv`; lock compatible DVC/GCS and
  MLflow versions without upgrading Flyte. Include required packages in the
  runtime image.
- Run focused regression tests, the full `uv run pytest` suite, lint, and existing
  CI checks before each PR. Verify the project's coverage target.
- Keep CI self-contained with temporary Git repositories, local DVC remotes, and
  a local MLflow server.
- Update architecture, standalone-training runbooks, and issue #7 delivery notes
  as implementation lands, distinguishing implemented capability from remaining
  upstream decoupling. This proposal itself completes no acceptance criteria.
- Merge Stage 1 after clean-workspace reproduction succeeds; merge Stage 2 after
  tracking-outage recovery succeeds.
- Record retrieval/publication overhead separately from fitting time. Neither
  stage claims to resolve the previously reported long-running pipeline.

## Design references

- [DVC remote storage](https://doc.dvc.org/user-guide/data-management/remote-storage)
- [DVC revision retrieval](https://doc.dvc.org/command-reference/get)
- [DVC retention and garbage collection](https://doc.dvc.org/command-reference/gc)
- [MLflow Tracking](https://mlflow.org/docs/latest/ml/tracking/)
- [Flyte MLflow integration](https://www.union.ai/docs/v2/flyte/integrations/mlflow/)

These references inform the proposal; dependency compatibility must be validated
against the repository lockfile when implementation begins.
