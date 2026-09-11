"""Remote Flyte tasks for the reproducible Junyi training workflow."""

from __future__ import annotations

import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from uuid import uuid4

import flyte
import numpy as np
from sqlalchemy import create_engine

from junyi_predictor.contracts import (
    FeatureSnapshot,
    PipelineRun,
    PreprocessedSnapshot,
)
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
from junyi_predictor.registry import register_model
from junyi_predictor.settings import Settings
from junyi_predictor.storage.artifacts import create_artifact_store

preprocess_env = flyte.TaskEnvironment(
    name="junyi-preprocess", image="auto", resources=flyte.Resources(memory="2Gi")
)
feature_env = flyte.TaskEnvironment(
    name="junyi-features", image="auto", resources=flyte.Resources(memory="4Gi")
)
train_env = flyte.TaskEnvironment(
    name="junyi-training", image="auto", resources=flyte.Resources(memory="4Gi")
)
pipeline_env = flyte.TaskEnvironment(
    name="junyi-pipeline",
    image="auto",
    resources=flyte.Resources(memory="1Gi"),
    depends_on=[preprocess_env, feature_env, train_env],
)


def _store_from_settings():
    settings = Settings.from_environment()
    return create_artifact_store(
        settings.artifact_backend, settings.artifact_root, settings.gcs_bucket
    )


def _feature_keys(run_id: str) -> tuple[str, str, str]:
    prefix = f"runs/{run_id}/features"
    return (
        f"{prefix}/log.parquet",
        f"{prefix}/concept_proficiency.npy",
        f"{prefix}/level4_proficiency.npy",
    )


def _preprocessed_keys(run_id: str) -> tuple[str, str, str]:
    prefix = f"runs/{run_id}/preprocessed"
    return (
        f"{prefix}/log.parquet",
        f"{prefix}/user.parquet",
        f"{prefix}/content.parquet",
    )


@preprocess_env.task(retries=1)
async def materialize_preprocessed(
    run_payload: dict,
) -> dict:
    """Load and persist a durable preprocessing result."""
    run = PipelineRun.model_validate(run_payload)
    settings = Settings.from_environment()
    engine = create_engine(settings.database_url)
    df_log, df_user, df_content = load_data_for_training(
        run.start_date, run.end_date, engine
    )
    preprocessed = preprocess_stage(df_log, df_user, df_content)
    preprocessed.log.assign(pipeline_run_id=run.run_id).to_sql(
        "processed_log", engine, if_exists="append", index=False
    )
    store = _store_from_settings()
    log_key, user_key, content_key = _preprocessed_keys(run.run_id)
    with tempfile.TemporaryDirectory(prefix="junyi-preprocessed-") as temp_dir:
        root = Path(temp_dir)
        log_path = root / "log.parquet"
        user_path = root / "user.parquet"
        content_path = root / "content.parquet"
        preprocessed.log.to_parquet(log_path, index=False)
        preprocessed.user.to_parquet(user_path, index=False)
        preprocessed.content.to_parquet(content_path, index=False)
        snapshot = PreprocessedSnapshot(
            run_id=run.run_id,
            log_uri=store.put_file(log_path, log_key),
            user_uri=store.put_file(user_path, user_key),
            content_uri=store.put_file(content_path, content_key),
            row_count=len(preprocessed.log),
        )
    store.put_json(
        snapshot.model_dump(mode="json"),
        f"runs/{run.run_id}/preprocessed_snapshot.json",
    )
    return snapshot.model_dump(mode="json")


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
    log_key, user_key, content_key = _preprocessed_keys(snapshot.run_id)
    return (
        store.get_file(log_key, root / "log.parquet"),
        store.get_file(user_key, root / "user.parquet"),
        store.get_file(content_key, root / "content.parquet"),
    )


@feature_env.task(retries=1)
async def materialize_feature_snapshot(preprocessed_payload: dict) -> dict:
    """Build and persist features from a validated preprocessing result."""
    snapshot = PreprocessedSnapshot.model_validate(preprocessed_payload)
    log_path, user_path, content_path = _load_preprocessed_files(snapshot)
    import pandas as pd

    featured = build_feature_stage(
        pd.read_parquet(log_path),
        pd.read_parquet(user_path),
        pd.read_parquet(content_path),
    )
    settings = Settings.from_environment()
    engine = create_engine(settings.database_url)
    featured.log.assign(pipeline_run_id=snapshot.run_id).to_sql(
        "feature_snapshot", engine, if_exists="append", index=False
    )
    store = _store_from_settings()
    log_key, concept_key, level4_key = _feature_keys(snapshot.run_id)
    with tempfile.TemporaryDirectory(prefix="junyi-features-") as temp_dir:
        root = Path(temp_dir)
        log_file = root / "log.parquet"
        concept_file = root / "concept_proficiency.npy"
        level4_file = root / "level4_proficiency.npy"
        featured.log.to_parquet(log_file, index=False)
        np.save(concept_file, featured.concept_proficiency)
        np.save(level4_file, featured.level4_proficiency)
        feature_snapshot = FeatureSnapshot(
            run_id=snapshot.run_id,
            log_uri=store.put_file(log_file, log_key),
            concept_matrix_uri=store.put_file(concept_file, concept_key),
            level4_matrix_uri=store.put_file(level4_file, level4_key),
            row_count=len(featured.log),
        )
    store.put_json(
        feature_snapshot.model_dump(mode="json"),
        f"runs/{snapshot.run_id}/feature_snapshot.json",
    )
    return feature_snapshot.model_dump(mode="json")


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
    log_key, concept_key, level4_key = _feature_keys(snapshot.run_id)
    return (
        store.get_file(log_key, root / "log.parquet"),
        store.get_file(concept_key, root / "concept.npy"),
        store.get_file(level4_key, root / "level4.npy"),
    )


@train_env.task(retries=1)
async def train_register(snapshot_payload: dict) -> dict:
    """Evaluate candidates, register the winner, and update approved-model metadata."""
    snapshot = FeatureSnapshot.model_validate(snapshot_payload)
    log_path, concept_path, level4_path = _load_snapshot_files(snapshot)
    import pandas as pd

    df_log = pd.read_parquet(log_path)
    split = split_training_data(
        df_log=df_log,
        m_concept_proficiency=np.load(concept_path),
        m_proficiency_level4=np.load(level4_path),
    )
    scaler = fit_min_max_scaler(split.X_train)
    X_train = scaler.transform(split.X_train)
    X_test = scaler.transform(split.X_test)
    models = {
        model_type: fit_model(X_train, split.y_train, model_type)
        for model_type in MODEL_TYPES
    }
    metrics = {
        model_type: {
            "train_score": float(model.score(X_train, split.y_train)),
            "test_score": float(model.score(X_test, split.y_test)),
        }
        for model_type, model in models.items()
    }
    winner = max(metrics, key=lambda model_type: metrics[model_type]["test_score"])
    registration = register_model(
        store=_store_from_settings(),
        run_id=snapshot.run_id,
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
async def training_pipeline(
    start_date: datetime | None = None,
    end_date: datetime | None = None,
    run_id: str = "",
) -> dict:
    """Compose the remotely executable preprocessing and training tasks."""
    resolved_end_date = end_date or datetime.utcnow()
    resolved_start_date = start_date or resolved_end_date - timedelta(days=7)
    resolved_run_id = (
        run_id or datetime.utcnow().strftime("%Y%m%dT%H%M%S") + "-" + uuid4().hex[:8]
    )
    pipeline_run = PipelineRun(
        run_id=resolved_run_id,
        start_date=resolved_start_date,
        end_date=resolved_end_date,
    )
    preprocessed = await materialize_preprocessed(pipeline_run.model_dump(mode="json"))
    feature_snapshot = await materialize_feature_snapshot(preprocessed)
    return await train_register(feature_snapshot)
