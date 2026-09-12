"""Remote Flyte tasks for the reproducible Junyi training workflow."""

from __future__ import annotations

import logging
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from uuid import uuid4

import flyte
import numpy as np
from sqlalchemy import create_engine, inspect
from sqlalchemy.exc import NoSuchTableError

from junyi_predictor.contracts import FeatureSnapshot, PipelineRun, PreprocessedSnapshot
from junyi_predictor.pipeline.constants import MODEL_TYPES
from junyi_predictor.pipeline.feature_engineering import build_feature_stage
from junyi_predictor.pipeline.preprocessing import (
    load_data_for_training,
    preprocess_stage,
)
from junyi_predictor.pipeline.training import (
    fit_min_max_scaler,
    fit_model,
    split_training_data,
)
from junyi_predictor.progress import operation, remote_logging_env, task_logging, timed
from junyi_predictor.registry import register_model
from junyi_predictor.settings import Settings
from junyi_predictor.storage.artifacts import create_artifact_store

logger = logging.getLogger(__name__)

preprocess_env = flyte.TaskEnvironment(
    name="junyi-preprocess",
    image="auto",
    resources=flyte.Resources(memory="2Gi"),
    env_vars=remote_logging_env(),
)
feature_env = flyte.TaskEnvironment(
    name="junyi-features",
    image="auto",
    resources=flyte.Resources(memory="4Gi"),
    env_vars=remote_logging_env(),
)
train_env = flyte.TaskEnvironment(
    name="junyi-training",
    image="auto",
    resources=flyte.Resources(memory="4Gi"),
    env_vars=remote_logging_env(),
)
pipeline_env = flyte.TaskEnvironment(
    name="junyi-pipeline",
    image="auto",
    resources=flyte.Resources(memory="1Gi"),
    depends_on=[preprocess_env, feature_env, train_env],
    env_vars=remote_logging_env(),
)


def _store_from_settings():
    settings = Settings.from_environment()
    return create_artifact_store(
        settings.artifact_backend, settings.artifact_root, settings.gcs_bucket
    )


def _feature_keys(training_run_id: str) -> tuple[str, str, str]:
    prefix = f"runs/{training_run_id}/features"
    return (
        f"{prefix}/log.parquet",
        f"{prefix}/concept_proficiency.npy",
        f"{prefix}/level4_proficiency.npy",
    )


def _preprocessed_keys(training_run_id: str) -> tuple[str, str, str]:
    prefix = f"runs/{training_run_id}/preprocessed"
    return (
        f"{prefix}/log.parquet",
        f"{prefix}/user.parquet",
        f"{prefix}/content.parquet",
    )


def _training_run_column(engine, table_name: str) -> str:
    """Use the new metadata name while keeping existing local output tables readable."""
    try:
        columns = {column["name"] for column in inspect(engine).get_columns(table_name)}
    except NoSuchTableError:
        columns = set()
    if "training_run_id" in columns or not columns:
        return "training_run_id"
    if "pipeline_run_id" in columns:
        return "pipeline_run_id"
    return "training_run_id"


@preprocess_env.task(retries=1)
@task_logging("preprocessing")
async def materialize_preprocessed(
    run_payload: dict,
) -> dict:
    """Load and persist a durable preprocessing result."""
    run = PipelineRun.model_validate(run_payload)
    store = _store_from_settings()
    snapshot_key = f"runs/{run.training_run_id}/preprocessed_snapshot.json"
    if store.exists(snapshot_key):
        raise ValueError(
            f"training_run_id already has a preprocessed snapshot: {run.training_run_id}"
        )
    settings = Settings.from_environment()
    engine = create_engine(settings.database_url)
    df_log, df_user, df_content = load_data_for_training(
        run.start_date, run.end_date, engine
    )
    preprocessed = preprocess_stage(df_log, df_user, df_content)
    with operation(
        "database.write",
        log=logger,
        table="processed_log",
        row_count=len(preprocessed.log),
    ):
        run_column = _training_run_column(engine, "processed_log")
        preprocessed.log.assign(**{run_column: run.training_run_id}).to_sql(
            "processed_log", engine, if_exists="append", index=False
        )
    log_key, user_key, content_key = _preprocessed_keys(run.training_run_id)
    with tempfile.TemporaryDirectory(prefix="junyi-preprocessed-") as temp_dir:
        root = Path(temp_dir)
        log_path = root / "log.parquet"
        user_path = root / "user.parquet"
        content_path = root / "content.parquet"
        with operation("preprocessing.serialize", log=logger):
            preprocessed.log.to_parquet(log_path, index=False)
            preprocessed.user.to_parquet(user_path, index=False)
            preprocessed.content.to_parquet(content_path, index=False)
        snapshot = PreprocessedSnapshot(
            training_run_id=run.training_run_id,
            log_uri=store.put_file(log_path, log_key),
            user_uri=store.put_file(user_path, user_key),
            content_uri=store.put_file(content_path, content_key),
            row_count=len(preprocessed.log),
        )
    store.put_json(
        snapshot.model_dump(mode="json"),
        snapshot_key,
    )
    return snapshot.model_dump(mode="json")


@timed("preprocessing.download")
def _load_preprocessed_files(
    snapshot: PreprocessedSnapshot,
) -> tuple[Path, Path, Path]:
    settings = Settings.from_environment()
    if settings.artifact_backend == "local":
        return (
            Path(snapshot.log_uri),
            Path(snapshot.user_uri),
            Path(snapshot.content_uri),
        )
    store = _store_from_settings()
    root = Path(tempfile.mkdtemp(prefix="junyi-features-"))
    log_key, user_key, content_key = _preprocessed_keys(snapshot.training_run_id)
    return (
        store.get_file(log_key, root / "log.parquet"),
        store.get_file(user_key, root / "user.parquet"),
        store.get_file(content_key, root / "content.parquet"),
    )


@feature_env.task(retries=1)
@task_logging("features")
async def materialize_feature_snapshot(preprocessed_payload: dict) -> dict:
    """Build and persist features from a validated preprocessing result."""
    snapshot = PreprocessedSnapshot.model_validate(preprocessed_payload)
    log_path, user_path, content_path = _load_preprocessed_files(snapshot)
    import pandas as pd

    with operation("features.load", log=logger):
        df_log = pd.read_parquet(log_path)
        df_user = pd.read_parquet(user_path)
        df_content = pd.read_parquet(content_path)
    featured = build_feature_stage(df_log, df_user, df_content)
    del df_log, df_user, df_content
    settings = Settings.from_environment()
    engine = create_engine(settings.database_url)
    with operation(
        "database.write",
        log=logger,
        table="feature_snapshot",
        row_count=len(featured.log),
    ):
        run_column = _training_run_column(engine, "feature_snapshot")
        featured.log.assign(**{run_column: snapshot.training_run_id}).to_sql(
            "feature_snapshot", engine, if_exists="append", index=False
        )
    store = _store_from_settings()
    log_key, concept_key, level4_key = _feature_keys(snapshot.training_run_id)
    with tempfile.TemporaryDirectory(prefix="junyi-features-") as temp_dir:
        root = Path(temp_dir)
        log_file = root / "log.parquet"
        concept_file = root / "concept_proficiency.npy"
        level4_file = root / "level4_proficiency.npy"
        with operation("features.serialize", log=logger):
            featured.log.to_parquet(log_file, index=False)
            np.save(concept_file, featured.concept_proficiency)
            np.save(level4_file, featured.level4_proficiency)
        feature_snapshot = FeatureSnapshot(
            training_run_id=snapshot.training_run_id,
            log_uri=store.put_file(log_file, log_key),
            concept_matrix_uri=store.put_file(concept_file, concept_key),
            level4_matrix_uri=store.put_file(level4_file, level4_key),
            row_count=len(featured.log),
        )
    store.put_json(
        feature_snapshot.model_dump(mode="json"),
        f"runs/{snapshot.training_run_id}/feature_snapshot.json",
    )
    return feature_snapshot.model_dump(mode="json")


@timed("features.download")
def _load_snapshot_files(snapshot: FeatureSnapshot) -> tuple[Path, Path, Path]:
    settings = Settings.from_environment()
    if settings.artifact_backend == "local":
        return (
            Path(snapshot.log_uri),
            Path(snapshot.concept_matrix_uri),
            Path(snapshot.level4_matrix_uri),
        )
    store = _store_from_settings()
    root = Path(tempfile.mkdtemp(prefix="junyi-training-"))
    log_key, concept_key, level4_key = _feature_keys(snapshot.training_run_id)
    return (
        store.get_file(log_key, root / "log.parquet"),
        store.get_file(concept_key, root / "concept.npy"),
        store.get_file(level4_key, root / "level4.npy"),
    )


@train_env.task(retries=1)
@task_logging("training")
async def train_register(snapshot_payload: dict) -> dict:
    """Evaluate candidates, register the winner, and update approved-model metadata."""
    snapshot = FeatureSnapshot.model_validate(snapshot_payload)
    log_path, concept_path, level4_path = _load_snapshot_files(snapshot)
    import pandas as pd

    with operation("training.load", log=logger):
        df_log = pd.read_parquet(log_path)
        concept = np.load(concept_path)
        level4 = np.load(level4_path)
    with operation("training.split", log=logger):
        split = split_training_data(
            df_log=df_log, m_concept_proficiency=concept, m_proficiency_level4=level4
        )
    del concept, level4
    with operation(
        "training.scale",
        log=logger,
        train_shape=split.X_train.shape,
        test_shape=split.X_test.shape,
        dtype=str(split.X_train.dtype),
    ):
        scaler = fit_min_max_scaler(split.X_train)
        X_train = scaler.transform(split.X_train)
        X_test = scaler.transform(split.X_test)
    models = {}
    for model_type in MODEL_TYPES:
        with operation("model.fit", log=logger, model_type=model_type):
            models[model_type] = fit_model(X_train, split.y_train, model_type)
    metrics = {}
    for model_type, model in models.items():
        with operation("model.evaluate", log=logger, model_type=model_type) as progress:
            metrics[model_type] = {
                "train_score": float(model.score(X_train, split.y_train)),
                "test_score": float(model.score(X_test, split.y_test)),
            }
            progress.update(**metrics[model_type])
    winner = max(metrics, key=lambda model_type: metrics[model_type]["test_score"])
    registration = register_model(
        store=_store_from_settings(),
        training_run_id=snapshot.training_run_id,
        model_type=winner,
        model=models[winner],
        scaler=scaler,
        metrics=metrics,
    )
    return registration.model_dump(mode="json")


@pipeline_env.task(
    triggers=flyte.Trigger(
        name="weekly-training",
        automation=flyte.Cron("0 3 * * 1"),
        description="Train and register the weekly Junyi model.",
    )
)
@task_logging("pipeline")
async def training_pipeline(
    start_date: datetime | None = None,
    end_date: datetime | None = None,
    training_run_id: str = "",
) -> dict:
    """Compose the remotely executable preprocessing and training tasks."""
    resolved_end_date = end_date or datetime.utcnow()
    resolved_start_date = start_date or resolved_end_date - timedelta(days=7)
    resolved_training_run_id = (
        training_run_id
        or datetime.utcnow().strftime("%Y%m%dT%H%M%S") + "-" + uuid4().hex[:8]
    )
    pipeline_run = PipelineRun(
        training_run_id=resolved_training_run_id,
        start_date=resolved_start_date,
        end_date=resolved_end_date,
    )
    logger.info(
        "Training interval selected",
        extra={
            "event": "pipeline.inputs",
            "start_date": resolved_start_date,
            "end_date": resolved_end_date,
        },
    )
    with operation("pipeline.preprocessing", log=logger, heartbeat=False):
        preprocessed = await materialize_preprocessed(
            pipeline_run.model_dump(mode="json")
        )
    with operation("pipeline.features", log=logger, heartbeat=False):
        feature_snapshot = await materialize_feature_snapshot(preprocessed)
    with operation("pipeline.training", log=logger, heartbeat=False):
        return await train_register(feature_snapshot)
