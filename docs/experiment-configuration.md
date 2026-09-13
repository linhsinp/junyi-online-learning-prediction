# Follow-up: experiment configuration and tracking

Status: proposed follow-up to the first stage-decoupling PR, tracked in
[issue #7](https://github.com/linhsinp/junyi-online-learning-prediction/issues/7).
The YAML interface below is not implemented by the first PR.

## Current configuration

Data scientists currently configure model hyperparameters in
`src/junyi_predictor/pipeline/training.py::fit_model`, candidate model types in
`pipeline/constants.py::MODEL_TYPES`, and scalar columns/matrix assembly in
`pipeline/experiments.py`. The training entrypoint already accepts a feature
snapshot, optional dates, and split fractions. The resolved estimator parameters
and selected feature layout are recorded in `training_metadata.json`.

## Next increment: validated experiment YAML

Provide a version-controlled YAML file so scientists can choose existing
features and configure multiple candidates without changing pipeline code.
An illustrative future file is:

```yaml
schema_version: v1
experiment: gradient-boosting-comparison

split:
  train: 0.70
  validation: 0.15

features:
  columns: [user_grade, level, v_upid_acc]
  matrix_blocks: [concept, level4]

candidates:
  - id: shallow
    model: GradientBoostingClassifier
    params:
      n_estimators: 100
      max_depth: 2
      learning_rate: 0.1
      random_state: 0
  - id: deeper
    model: GradientBoostingClassifier
    params:
      n_estimators: 100
      max_depth: 4
      learning_rate: 0.1
      random_state: 0
```

Each candidate must have a unique ID: metrics keyed only by estimator type would
overwrite results when two candidates use the same model class. Explicit
candidates are the initial tuning interface; automatic grid/random search or an
optimizer can follow separately.

Keep existing defaults when no file is supplied. Validate the configuration
before expensive loading/fitting: supported models and parameters, unique trial
IDs, available features, and valid split fractions. Reject unknown fields and
invalid selections with actionable errors. Resolve defaults and any command-line
overrides once; document their precedence. Pass the resolved configuration as a
Flyte input, so remote workers do not depend on a scientist's local YAML path.
Persist that resolved configuration, its schema version, and its hash alongside
every run's dataset fingerprints and split metadata.

Feature selection means selecting columns or matrix blocks already present in
the pinned snapshot. A change to feature computation requires a new feature
definition/publication through the feature engineering component. Legacy v1
matrices lack semantic concept-to-column mappings, so named selection of
individual concept columns must wait for versioned mappings. Target and event
timestamp fields remain required for evaluation even when they are not predictors.

Fit candidates and learned transformations on training rows, compare candidates
on validation, and evaluate only the selected winner on held-out test. Standalone
experiments continue to register without automatic promotion.

### Acceptance and tests

- [ ] Run two parameterizations of one estimator with distinct IDs and retain
  both sets of parameters and validation metrics.
- [ ] Change selected scalar columns/matrix blocks without upstream execution;
  verify feature order, row alignment, and train-only transformation fitting.
- [ ] Reject malformed YAML, unsupported parameters, duplicate IDs, unavailable
  features, empty predictor selection, and invalid split fractions before fitting.
- [ ] Preserve current behavior when no YAML is supplied; test documented
  override precedence and stable hashing of the resolved configuration.
- [ ] Run with the resolved configuration after the original YAML file is
  unavailable to the worker; verify complete configuration and dataset lineage.
- [ ] Keep test rows out of candidate selection and record a test score only for
  the selected winner.

## Later increment: MLflow experiment comparison

Flyte executes the workflows and supports custom
[task reports](https://www.union.ai/docs/v2/flyte/user-guide/tasks/task-programming/reports/).
[MLflow Tracking](https://mlflow.org/docs/latest/ml/tracking) supplies a UI and API
for comparing parameters, metrics, code versions, and artifacts across runs.
Flyte documents an
[MLflow integration](https://www.union.ai/docs/v2/flyte/integrations/mlflow/);
compatibility with this repository's pinned environment must be checked when
implementing it. Neither MLflow nor that plugin is configured in this repo today.

Use the YAML as the experiment definition and MLflow as an optional consumer of
the resolved configuration and results. Suggested organization: one parent run
per experiment and one child run per candidate, tagged with Flyte identifiers,
candidate ID, dataset identity, and configuration hash. Record the held-out test
score only on the selected candidate. Keep raw learner records out of tracking.

Do not require MLflow to use YAML configuration. Retain the existing artifact
store and model registration contract; adding experiment tracking must not
silently introduce another source of truth for approved models. Define and test
tracking failure/retry behavior before enabling it for scheduled runs.
