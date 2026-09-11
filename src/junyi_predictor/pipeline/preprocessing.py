"""Preprocessing stage contracts and pure transformations."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from sqlalchemy import Engine

from junyi_predictor.paths import (
    CONTENT_FILE,
    CURATED_LOG_DIR,
    EXPERIMENT_DATA_DIR,
    LOG_FILE,
    OUTPUT_DATA_DIR,
    RAW_DATA_DIR,
    TEST_DATA_DIR,
    USER_FILE,
)

VARS_REDUNDANT = ["total_sec_taken", "is_hint_used", "is_downgrade", "is_upgrade"]
PATH_INPUT = str(RAW_DATA_DIR)
PATH_OUTPUT = str(OUTPUT_DATA_DIR)
PATH_EXPERIMENT = str(EXPERIMENT_DATA_DIR)
PATH_TEST = str(TEST_DATA_DIR)

FILE_LOG_FULL = str(LOG_FILE)
FILE_USER = str(USER_FILE)
FILE_CONTENT = str(CONTENT_FILE)


@dataclass(frozen=True)
class PreprocessStageOutput:
    """Explicit output contract for the preprocessing stage."""

    log: pd.DataFrame
    user: pd.DataFrame
    content: pd.DataFrame


def load_raw_dataframes(
    path_log_full: str, path_user: str, path_content: str
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load raw CSV inputs used by the preprocessing stage."""
    log_dtypes = {
        "timestamp_TW": "object",
        "uuid": "category",
        "ucid": "category",
        "upid": "category",
        "problem_number": "int16",
        "exercise_problem_repeat_session": "int16",
        "is_correct": "boolean",
        "total_sec_taken": "int16",
        "total_attempt_cnt": "int16",
        "used_hint_cnt": "int16",
        "is_hint_used": "boolean",
        "level": "int8",
    }
    user_dtype = {"uuid": "category", "gender": "category", "user_grade": "int8"}
    content_dtype = {
        "ucid": "category",
        "level4_id": "category",
        "difficulty": "category",
        "learning_stage": "category",
    }

    df_log = pd.read_csv(path_log_full, dtype=log_dtypes)
    df_user = pd.read_csv(path_user, dtype=user_dtype)
    df_content = pd.read_csv(path_content, dtype=content_dtype)

    assert not df_log.empty, "Log data is empty."
    assert not df_user.empty, "User data is empty."
    assert not df_content.empty, "Content data is empty."

    return df_log, df_user, df_content


def load_data_for_training(
    start_date: datetime,
    end_date: datetime,
    sqlmodel_engine: Engine,
    curated_log_root: Path = CURATED_LOG_DIR,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load date-filtered event history from Parquet and dimensions from PostgreSQL."""
    start = _as_naive_utc(start_date)
    end = _as_naive_utc(end_date)
    partitions: list[Path] = []
    partition_month = datetime(start.year, start.month, 1)
    while partition_month < end:
        path = (
            curated_log_root
            / f"year={partition_month.year}"
            / f"month={partition_month.month:02d}"
        )
        if path.exists():
            partitions.append(path)
        if partition_month.month == 12:
            partition_month = datetime(partition_month.year + 1, 1, 1)
        else:
            partition_month = datetime(
                partition_month.year, partition_month.month + 1, 1
            )
    frames = [pd.read_parquet(path) for path in partitions]
    if not frames:
        raise FileNotFoundError(
            f"No curated log partitions found under {curated_log_root}"
        )
    df_log = pd.concat(frames, ignore_index=True)
    df_log = df_log.loc[
        (df_log["timestamp_TW"] >= start) & (df_log["timestamp_TW"] < end)
    ].copy()
    selected_uuid = df_log["uuid"].unique().tolist()
    df_user = pd.read_sql(
        "SELECT * FROM user_profile WHERE uuid = ANY(%(selected_uuid)s)",
        sqlmodel_engine,
        params={"selected_uuid": selected_uuid},
    )
    df_content = pd.read_sql("SELECT * FROM info_content;", sqlmodel_engine)
    return df_log, df_user, df_content


def _as_naive_utc(value: datetime) -> datetime:
    """Normalize Flyte's UTC-aware inputs to Parquet's naive UTC timestamps."""
    if value.tzinfo is None:
        return value
    return value.astimezone(timezone.utc).replace(tzinfo=None)


def preprocess_log_frame(df_log: pd.DataFrame, df_user: pd.DataFrame) -> pd.DataFrame:
    """Merge user attributes, normalize gender flags, sort rows, and drop leakage columns."""
    df_log = pd.merge(
        df_log.copy(),
        df_user[["uuid", "user_grade", "gender"]],
        on="uuid",
        how="left",
    )
    df_log = df_log.sort_values(["timestamp_TW", "uuid", "upid"]).reset_index(drop=True)

    df_log.fillna(value={"gender": "unspecified"}, inplace=True)
    gender_flags = pd.get_dummies(df_log["gender"])
    gender_flags = gender_flags.reindex(
        columns=["female", "male", "unspecified"], fill_value=0
    )
    df_log = pd.concat([df_log.drop(columns="gender"), gender_flags], axis=1)

    df_log["level"] = (
        df_log["level"]
        + df_log["is_downgrade"].fillna(False).astype(int)
        - df_log["is_upgrade"].fillna(False).astype(int)
    ).astype("int8")

    return df_log.drop(columns=VARS_REDUNDANT)


def preprocess_stage(
    df_log: pd.DataFrame, df_user: pd.DataFrame, df_content: pd.DataFrame
) -> PreprocessStageOutput:
    """Run the preprocessing stage and return an explicit stage contract."""
    processed_log = preprocess_log_frame(df_log=df_log, df_user=df_user)
    return PreprocessStageOutput(
        log=processed_log, user=df_user.copy(), content=df_content.copy()
    )
