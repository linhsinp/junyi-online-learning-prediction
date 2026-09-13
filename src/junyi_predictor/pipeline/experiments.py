"""Shared snapshot selection, chronological evaluation, and run registration."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import datetime
from importlib.metadata import version

import numpy as np
import pandas as pd

from junyi_predictor.contracts import ModelRegistration
from junyi_predictor.pipeline.constants import MODEL_TYPES
from junyi_predictor.pipeline.training import fit_min_max_scaler, fit_model
from junyi_predictor.progress import operation
from junyi_predictor.registry import register_model, validate_training_run_id
from junyi_predictor.storage.artifacts import ArtifactStore
from junyi_predictor.storage.feature_snapshots import load_feature_snapshot

logger = logging.getLogger(__name__)
SCALAR_COLUMNS = (
    "user_grade",
    "female",
    "male",
    "unspecified",
    "v_upid_acc",
    "level",
    "problem_number",
    "exercise_problem_repeat_session",
)


@dataclass(frozen=True)
class EvaluationSplit:
    """Aligned partitions plus sufficient metadata to repeat their selection."""

    X_train: np.ndarray
    y_train: np.ndarray
    X_validation: np.ndarray
    y_validation: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    metadata: dict


def _utc(value: datetime | None) -> pd.Timestamp | None:
    if value is None:
        return None
    timestamp = pd.Timestamp(value)
    if pd.isna(timestamp):
        raise ValueError("Date bounds must be valid timestamps")
    return (
        timestamp.tz_localize("UTC")
        if timestamp.tzinfo is None
        else timestamp.tz_convert("UTC")
    )


def select_and_split(
    log: pd.DataFrame,
    concept: np.ndarray,
    level4: np.ndarray,
    *,
    expected_rows: int,
    start_date: datetime | None = None,
    end_date: datetime | None = None,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
) -> EvaluationSplit:
    """Filter the complete input, then split without crossing timestamp groups."""
    if (
        not math.isfinite(train_fraction)
        or not math.isfinite(validation_fraction)
        or train_fraction <= 0
        or validation_fraction <= 0
        or train_fraction + validation_fraction >= 1
    ):
        raise ValueError("Split fractions must be positive with sum below one")
    start, end = _utc(start_date), _utc(end_date)
    if start is not None and end is not None and start >= end:
        raise ValueError("start_date must precede end_date")
    required = {*SCALAR_COLUMNS, "timestamp_TW", "is_correct"}
    if not required.issubset(log.columns):
        raise ValueError("Feature log is missing required training columns")
    if len(log) != expected_rows:
        raise ValueError("Feature log row count does not match manifest")
    for matrix in (concept, level4):
        if matrix.ndim != 2 or matrix.shape[0] != expected_rows:
            raise ValueError("Feature matrices must be two-dimensional and row-aligned")
        if not np.issubdtype(matrix.dtype, np.number) or not np.isfinite(matrix).all():
            raise ValueError("Feature matrices must contain finite numeric inputs")
    timestamps = pd.to_datetime(log["timestamp_TW"], utc=True, errors="coerce")
    if timestamps.isna().any() or not timestamps.is_monotonic_increasing:
        raise ValueError("Feature timestamps must be valid and chronologically ordered")
    try:
        scalars = log[list(SCALAR_COLUMNS)].to_numpy(dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "Scalar features must contain finite numeric inputs"
        ) from error
    if not np.isfinite(scalars).all():
        raise ValueError("Scalar features must contain finite numeric inputs")
    if not log["is_correct"].isin([True, False]).all():
        raise ValueError("Targets must contain only non-null binary values")

    mask = np.ones(len(log), dtype=bool)
    if start is not None:
        mask &= (timestamps >= start).to_numpy()
    if end is not None:
        mask &= (timestamps < end).to_numpy()
    selected_timestamps = timestamps.loc[mask].reset_index(drop=True)
    n = len(selected_timestamps)
    cuts = [
        math.floor(n * train_fraction),
        math.floor(n * (train_fraction + validation_fraction)),
    ]
    for index, cut in enumerate(cuts):
        if 0 < cut < n:
            cuts[index] = int(
                selected_timestamps.searchsorted(
                    selected_timestamps.iloc[cut - 1], side="right"
                )
            )
    first, second = cuts
    if not 0 < first < second < n:
        raise ValueError(
            "Selection must produce three nonempty chronological partitions"
        )
    y = log.loc[mask, "is_correct"].to_numpy(dtype=bool)
    if np.unique(y[:first]).size != 2:
        raise ValueError("Training partition must contain both target classes")
    X = np.concatenate((scalars[mask], concept[mask], level4[mask]), axis=1)
    partitions = {}
    for name, left, right in (
        ("train", 0, first),
        ("validation", first, second),
        ("test", second, n),
    ):
        partitions[name] = {
            "row_count": right - left,
            "start_position": left,
            "end_position_exclusive": right,
            "first_timestamp": selected_timestamps.iloc[left].isoformat(),
            "last_timestamp": selected_timestamps.iloc[right - 1].isoformat(),
        }
    metadata = {
        "requested_start_date": start.isoformat() if start is not None else None,
        "requested_end_date": end.isoformat() if end is not None else None,
        "selected_row_count": n,
        "fractions": {
            "train": train_fraction,
            "validation": validation_fraction,
            "test": 1 - train_fraction - validation_fraction,
        },
        "partitions": partitions,
        "scalar_columns": list(SCALAR_COLUMNS),
        "matrix_blocks": [
            {
                "name": "concept",
                "columns": concept.shape[1],
                "dtype": str(concept.dtype),
                "shape": list(concept.shape),
            },
            {
                "name": "level4",
                "columns": level4.shape[1],
                "dtype": str(level4.dtype),
                "shape": list(level4.shape),
            },
        ],
        "assembled_dtype": str(X.dtype),
    }
    return EvaluationSplit(
        X[:first],
        y[:first],
        X[first:second],
        y[first:second],
        X[second:],
        y[second:],
        metadata,
    )


def evaluate_candidates(
    split: EvaluationSplit,
) -> tuple[object, object, str, dict, dict]:
    """Select on validation accuracy and evaluate only the winner on test data."""
    with operation(
        "training.scale",
        log=logger,
        train_shape=split.X_train.shape,
        validation_shape=split.X_validation.shape,
        test_shape=split.X_test.shape,
        dtype=str(split.X_train.dtype),
    ):
        scaler = fit_min_max_scaler(split.X_train)
        X_train = scaler.transform(split.X_train)
        X_validation = scaler.transform(split.X_validation)
    winner, winning_model, best_score = "", None, -math.inf
    metrics, configurations = {}, {}
    for model_type in MODEL_TYPES:
        with operation("model.fit", log=logger, model_type=model_type):
            model = fit_model(X_train, split.y_train, model_type)
        configurations[model_type] = model.get_params(deep=False)
        with operation("model.evaluate", log=logger, model_type=model_type) as progress:
            metrics[model_type] = {
                "train_score": float(model.score(X_train, split.y_train)),
                "validation_score": float(
                    model.score(X_validation, split.y_validation)
                ),
            }
            progress.update(**metrics[model_type])
        score = metrics[model_type]["validation_score"]
        if score > best_score:
            winner, winning_model, best_score = model_type, model, score
    logger.info(
        "Candidate selected using validation accuracy",
        extra={
            "event": "training.selected",
            "model_type": winner,
            "validation_score": best_score,
        },
    )
    with operation("model.test", log=logger, model_type=winner) as progress:
        X_test = scaler.transform(split.X_test)
        metrics[winner]["test_score"] = float(winning_model.score(X_test, split.y_test))
        progress.update(test_score=metrics[winner]["test_score"])
    return winning_model, scaler, winner, metrics, configurations


def train_snapshot(
    store: ArtifactStore,
    feature_snapshot_key: str,
    training_run_id: str,
    *,
    start_date: datetime | None = None,
    end_date: datetime | None = None,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
    promote: bool = False,
) -> ModelRegistration:
    """Train from retained feature artifacts without database or upstream access."""
    validate_training_run_id(training_run_id)
    if store.exists(f"models/{training_run_id}/manifest.json"):
        raise ValueError("training_run_id already has a registered model")
    with load_feature_snapshot(store, feature_snapshot_key) as loaded:
        with operation("training.load", log=logger):
            log = pd.read_parquet(loaded.log_path)
            concept = np.load(loaded.concept_path, allow_pickle=False)
            level4 = np.load(loaded.level4_path, allow_pickle=False)
        with operation("training.split", log=logger):
            split = select_and_split(
                log,
                concept,
                level4,
                expected_rows=loaded.snapshot.row_count,
                start_date=start_date,
                end_date=end_date,
                train_fraction=train_fraction,
                validation_fraction=validation_fraction,
            )
        metadata = {
            "training_run_id": training_run_id,
            "feature_snapshot_key": feature_snapshot_key,
            "source_training_run_id": loaded.snapshot.training_run_id,
            "source_manifest": loaded.manifest,
            "input_sha256": loaded.fingerprints,
            "evaluation_protocol": "chronological-train-validation-test-v1",
            "selection": split.metadata,
            "library_versions": {
                name: version(name) for name in ("numpy", "pandas", "scikit-learn")
            },
        }
    del log, concept, level4
    logger.info(
        "Feature snapshot selected for training",
        extra={
            "event": "training.inputs",
            "feature_snapshot_key": feature_snapshot_key,
            "source_training_run_id": metadata["source_training_run_id"],
            "start_date": split.metadata["requested_start_date"],
            "end_date": split.metadata["requested_end_date"],
            "selected_row_count": split.metadata["selected_row_count"],
            "partition_counts": {
                name: value["row_count"]
                for name, value in split.metadata["partitions"].items()
            },
        },
    )
    model, scaler, winner, metrics, configurations = evaluate_candidates(split)
    metadata["candidate_configurations"] = configurations
    return register_model(
        store,
        training_run_id,
        winner,
        model,
        scaler,
        metrics,
        training_metadata=metadata,
        promote=promote,
    )
