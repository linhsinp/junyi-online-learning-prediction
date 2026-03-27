"""Preprocessing stage contracts and pure transformations."""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime

import pandas as pd
from sqlalchemy import Engine

VARS_REDUNDANT = ["total_sec_taken", "is_hint_used", "is_downgrade", "is_upgrade"]
PATH_INPUT = "data/raw"
PATH_OUTPUT = "data/output"
PATH_EXPERIMENT = "data/experiment"
PATH_TEST = "data/test"

FILE_LOG_FULL = os.path.join(PATH_INPUT, "Log_Problem.csv")
FILE_USER = os.path.join(PATH_INPUT, "Info_UserData.csv")
FILE_CONTENT = os.path.join(PATH_INPUT, "Info_Content.csv")


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


def load_data_from_database(
    start_date: datetime, end_date: datetime, sqlmodel_engine: Engine
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load log, user, and content frames from the database."""
    log_query = """
        SELECT *
        FROM log_problem
        WHERE "timestamp_TW" >= %(start)s
        AND "timestamp_TW" < %(end)s
        """
    df_log = pd.read_sql(
        log_query, sqlmodel_engine, params={"start": start_date, "end": end_date}
    )

    selected_uuid = df_log["uuid"].unique().tolist()
    user_query = """
        SELECT *
        FROM user_profile
        WHERE uuid = ANY(%(selected_uuid)s)
        """
    df_user = pd.read_sql(
        user_query, sqlmodel_engine, params={"selected_uuid": selected_uuid}
    )

    df_content = pd.read_sql("SELECT * FROM info_content;", sqlmodel_engine)
    return df_log, df_user, df_content


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
        + df_log["is_downgrade"].fillna(0).astype(int)
        - df_log["is_upgrade"].fillna(0).astype(int)
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
