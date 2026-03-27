import os
import shutil
from datetime import datetime

from dotenv import load_dotenv
from sqlalchemy import create_engine

import flyte
from junyi_predictor.pipeline.feature_engineering import build_feature_stage
from junyi_predictor.pipeline.preprocessing import (
    load_data_from_database,
    preprocess_stage,
)
from junyi_predictor.pipeline.training import (
    apply_min_max_transformation,
    load_feature_matrices,
    load_parquet_dataframe,
    split_training_data,
    train_and_evaluate_model,
)
from junyi_predictor.storage.gcs import download_data_to_tmp
from orchestration.constants import MODEL_TYPES

load_dotenv()

env = flyte.TaskEnvironment(
    name="junyi-predictor",
    image="auto",
    resources=flyte.Resources(memory="4Gi"),
)


def _train_all_models(
    X_train,
    y_train,
    X_test,
    y_test,
) -> dict:
    metrics: dict[str, dict] = {}
    for model_type in MODEL_TYPES:
        metrics[model_type] = train_and_evaluate_model(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            model_type=model_type,
        )
    return metrics


@env.task
async def preprocess_from_database(
    start_date: datetime = datetime(2019, 6, 1),
    end_date: datetime = datetime(2019, 6, 10),
) -> dict:
    """Load source data from PostgreSQL and return a lightweight preprocessing summary."""
    db_url = os.environ["DATABASE_URL"]
    sqlmodel_engine = create_engine(db_url)
    df_log, df_user, df_content = load_data_from_database(
        start_date=start_date,
        end_date=end_date,
        sqlmodel_engine=sqlmodel_engine,
    )
    preprocessed = preprocess_stage(
        df_log=df_log, df_user=df_user, df_content=df_content
    )
    return {
        "log_rows": int(preprocessed.log.shape[0]),
        "user_rows": int(preprocessed.user.shape[0]),
        "content_rows": int(preprocessed.content.shape[0]),
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
    }


@env.task
async def full_pipeline(
    start_date: datetime = datetime(2019, 6, 1),
    end_date: datetime = datetime(2019, 6, 10),
    num_samples: int | None = 50000,
) -> dict:
    """Run the end-to-end training flow using PostgreSQL as the source of truth."""
    db_url = os.environ["DATABASE_URL"]
    sqlmodel_engine = create_engine(db_url)
    df_log, df_user, df_content = load_data_from_database(
        start_date=start_date,
        end_date=end_date,
        sqlmodel_engine=sqlmodel_engine,
    )
    preprocessed = preprocess_stage(
        df_log=df_log, df_user=df_user, df_content=df_content
    )
    feature_output = build_feature_stage(
        df_log=preprocessed.log,
        df_user=preprocessed.user,
        df_content=preprocessed.content,
    )
    split = split_training_data(
        df_log=feature_output.log,
        m_concept_proficiency=feature_output.concept_proficiency,
        m_proficiency_level4=feature_output.level4_proficiency,
        num_samples=num_samples,
    )
    X_train, X_test = apply_min_max_transformation(split.X_train, split.X_test)
    return _train_all_models(X_train, split.y_train, X_test, split.y_test)


@env.task
async def train_from_gcs(
    prefix: str = "feature_store",
    local_dir: str = "/tmp/data",
) -> dict:
    """Download engineered features from GCS and train all configured models."""
    download_data_to_tmp(prefix=prefix, local_dir=local_dir)
    try:
        feature_root = os.path.join(local_dir)
        df_log = load_parquet_dataframe(
            os.path.join(feature_root, "df_log_with_upid_acc.parquet.gzip")
        )
        concept_matrix, level4_matrix = load_feature_matrices(
            os.path.join(feature_root, "m_concept_proficiency.npz"),
            os.path.join(feature_root, "m_proficiency_level4.npz"),
        )
        split = split_training_data(
            df_log=df_log,
            m_concept_proficiency=concept_matrix,
            m_proficiency_level4=level4_matrix,
        )
        X_train, X_test = apply_min_max_transformation(split.X_train, split.X_test)
        return _train_all_models(X_train, split.y_train, X_test, split.y_test)
    finally:
        shutil.rmtree(local_dir, ignore_errors=True)
