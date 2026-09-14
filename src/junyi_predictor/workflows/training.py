"""Remote Flyte tasks for the reproducible Junyi training workflow."""

from __future__ import annotations

import logging
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from uuid import uuid4

import flyte
import numpy as np
from sqlalchemy import create_engine

from junyi_predictor.contracts import FeatureSnapshot, PipelineRun, PreprocessedSnapshot
from junyi_predictor.pipeline.experiments import train_snapshot
from junyi_predictor.pipeline.feature_engineering import build_feature_stage
from junyi_predictor.pipeline.preprocessing import (
    load_data_for_training,
    preprocess_stage,
)
from junyi_predictor.progress import operation, remote_logging_env, task_logging, timed
from junyi_predictor.settings import ArtifactSettings, DataLakeSettings, Settings
from junyi_predictor.storage.artifacts import create_artifact_store
from junyi_predictor.storage.data_lake import resolve_curated_log_root

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
training_only_env = flyte.TaskEnvironment(
    name="junyi-training-only",
    image="auto",
    resources=flyte.Resources(memory="1Gi"),
    depends_on=[train_env],
    env_vars=remote_logging_env(),
)


def _store_from_settings():
    settings = ArtifactSettings.from_environment()
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
    data_lake_settings = DataLakeSettings.from_environment()
    if data_lake_settings.backend == "local":
        df_log, df_user, df_content = load_data_for_training(
            run.start_date, run.end_date, engine
        )
    else:
        with tempfile.TemporaryDirectory(prefix="junyi-curated-") as temp_dir:
            curated_log_root = resolve_curated_log_root(
                data_lake_settings,
                run.start_date,
                run.end_date,
                Path(temp_dir),
            )
            df_log, df_user, df_content = load_data_for_training(
                run.start_date, run.end_date, engine, curated_log_root
            )
    preprocessed = preprocess_stage(df_log, df_user, df_content)
    with operation(
        "database.write",
        log=logger,
        table="processed_log",
        row_count=len(preprocessed.log),
    ):
        preprocessed.log.assign(training_run_id=run.training_run_id).to_sql(
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
        featured.log.assign(training_run_id=snapshot.training_run_id).to_sql(
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


@train_env.task(retries=1)
@task_logging("training")
async def train_register(
    snapshot_payload: dict,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
) -> dict:
    """Train the combined workflow's published snapshot and promote its winner."""
    snapshot = FeatureSnapshot.model_validate(snapshot_payload)
    registration = train_snapshot(
        _store_from_settings(),
        f"runs/{snapshot.training_run_id}/feature_snapshot.json",
        snapshot.training_run_id,
        train_fraction=train_fraction,
        validation_fraction=validation_fraction,
        promote=True,
    )
    return registration.model_dump(mode="json")


@train_env.task(retries=1)
@task_logging("training")
async def train_feature_experiment(
    training_run_id: str,
    feature_snapshot_key: str,
    start_date: datetime | None = None,
    end_date: datetime | None = None,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
) -> dict:
    """Register an independent experiment without promoting it."""
    registration = train_snapshot(
        _store_from_settings(),
        feature_snapshot_key,
        training_run_id,
        start_date=start_date,
        end_date=end_date,
        train_fraction=train_fraction,
        validation_fraction=validation_fraction,
    )
    return registration.model_dump(mode="json")


@training_only_env.task
@task_logging("pipeline")
async def train_from_features(
    feature_snapshot_key: str,
    training_run_id: str = "",
    start_date: datetime | None = None,
    end_date: datetime | None = None,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
) -> dict:
    """Select a retained feature snapshot and train without upstream execution."""
    return await train_feature_experiment(
        training_run_id,
        feature_snapshot_key,
        start_date,
        end_date,
        train_fraction,
        validation_fraction,
    )


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
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
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
        return await train_register(
            feature_snapshot, train_fraction, validation_fraction
        )
