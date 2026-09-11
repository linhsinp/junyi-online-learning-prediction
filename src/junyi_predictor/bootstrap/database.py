"""Helpers for seeding PostgreSQL from local raw artifacts."""

from __future__ import annotations

import math
from datetime import date, datetime
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd
from pydantic import ValidationError
from sqlalchemy.engine import Engine
from sqlmodel import Field, Session, SQLModel, create_engine

from junyi_predictor.paths import CONTENT_FILE, LOG_FILE, USER_FILE
from junyi_predictor.pipeline.preprocessing import load_raw_dataframes

DEFAULT_ENGINE_URL = "postgresql://postgres:postgres@localhost:30001/postgres"


class DifficultyEnum(str, Enum):
    easy = "easy"
    hard = "hard"
    normal = "normal"
    unset = "unset"


class LearningStageEnum(str, Enum):
    elementary = "elementary"
    junior = "junior"
    senior = "senior"


ENUM_MAP_CONTENT = {
    "difficulty": DifficultyEnum,
    "learning_stage": LearningStageEnum,
}


class InfoContent(SQLModel, table=True):
    __tablename__ = "info_content"
    __table_args__ = {"extend_existing": True}

    ucid: str = Field(primary_key=True)
    content_pretty_name: str
    content_kind: str
    difficulty: DifficultyEnum
    subject: str
    learning_stage: LearningStageEnum
    level1_id: str
    level2_id: str
    level3_id: str
    level4_id: str


class UserProfile(SQLModel, table=True):
    __tablename__ = "user_profile"
    __table_args__ = {"extend_existing": True}

    uuid: str = Field(primary_key=True)
    gender: str | None = Field(default=None)
    points: int
    badges_cnt: int
    first_login_date_TW: date
    user_grade: int
    user_city: str
    has_teacher_cnt: int
    is_self_coach: bool
    has_student_cnt: int
    belongs_to_class_cnt: int
    has_class_cnt: int


class BoolObjectEnum(str, Enum):
    true = "True"
    false = "False"
    none = "None"


ENUM_MAP_LOG_PROBLEM = {
    "is_downgrade": BoolObjectEnum,
    "is_upgrade": BoolObjectEnum,
}


class LogProblem(SQLModel, table=True):
    __tablename__ = "log_problem"
    __table_args__ = {"extend_existing": True}

    timestamp_TW: datetime | None = Field(primary_key=True)
    uuid: str | None = Field(primary_key=True)
    ucid: str | None = Field(foreign_key="info_content.ucid")
    upid: str | None = Field(primary_key=True)
    problem_number: int | None = Field(default=None)
    exercise_problem_repeat_session: int | None = Field(default=None)
    is_correct: bool | None = Field(default=None)
    total_sec_taken: int | None = Field(default=None)
    total_attempt_cnt: int | None = Field(default=None)
    used_hint_cnt: int | None = Field(default=None)
    is_hint_used: bool | None = Field(default=None)
    is_downgrade: BoolObjectEnum | None = Field(default=None)
    is_upgrade: BoolObjectEnum | None = Field(default=None)
    level: int | None = Field(default=None)


ENUM_MAPS = {
    "info_content": ENUM_MAP_CONTENT,
    "log_problem": ENUM_MAP_LOG_PROBLEM,
}


def to_enum_aware_dict(
    row: dict[str, Any], enum_map: dict[str, type[Enum]]
) -> dict[str, Any]:
    """Convert selected fields into Enum instances while keeping null-like values nullable."""

    def safe_enum(enum_cls: type[Enum], value: Any) -> Enum | None:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return None
        try:
            return enum_cls(value)
        except ValueError:
            return None

    return {
        key: safe_enum(enum_map[key], value)
        if key in enum_map
        else (None if isinstance(value, float) and math.isnan(value) else value)
        for key, value in row.items()
    }


def validate_with_sqlmodel(
    df: pd.DataFrame,
    model_class: type[SQLModel],
    enum_maps: dict[str, dict[str, type[Enum]]] = ENUM_MAPS,
) -> pd.DataFrame:
    """Validate DataFrame rows against a SQLModel schema with optional Enum coercion."""
    table_name = getattr(model_class, "__tablename__", None)
    enum_map = enum_maps.get(table_name)

    valid_data: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        try:
            row_dict = row.to_dict()
            if enum_map:
                row_dict = to_enum_aware_dict(row_dict, enum_map)
            record = model_class(**row_dict)
            valid_data.append(record.model_dump())
        except ValidationError as error:
            print(f"Validation failed: {error}")
    return pd.DataFrame(valid_data)


def create_table_from_dataframe(
    df: pd.DataFrame,
    model_class: type[SQLModel],
    engine: Engine,
    enum_maps: dict[str, dict[str, type[Enum]]] = ENUM_MAPS,
) -> None:
    """Create a table and insert validated rows from a DataFrame."""
    validated_df = validate_with_sqlmodel(df, model_class, enum_maps)
    SQLModel.metadata.create_all(engine)

    with Session(engine) as session:
        objects = [model_class(**row) for row in validated_df.to_dict(orient="records")]
        session.add_all(objects)
        session.commit()


def chunked_upload_with_validation(
    df: pd.DataFrame,
    model_class: type[SQLModel],
    engine: Engine,
    table_name: str,
    chunk_size: int = 100_000,
    enum_maps: dict[str, dict[str, type[Enum]]] = ENUM_MAPS,
) -> None:
    """Validate and upload a large DataFrame in chunks."""
    total_chunks = math.ceil(len(df) / chunk_size)

    for offset in range(0, len(df), chunk_size):
        chunk = df.iloc[offset : offset + chunk_size]
        validated_chunk = validate_with_sqlmodel(chunk, model_class, enum_maps)
        validated_chunk.to_sql(
            table_name, engine, if_exists="append", index=False, method="multi"
        )
    print(f"Uploaded {total_chunks} chunk(s) to {table_name}.")


def seed_database_from_raw_files(engine_url: str = DEFAULT_ENGINE_URL) -> None:
    """Load raw local artifacts and seed the PostgreSQL tables used by the workflows."""
    engine = create_engine(engine_url)
    df_log, df_user, df_content = load_raw_dataframes(
        str(LOG_FILE), str(USER_FILE), str(CONTENT_FILE)
    )

    create_table_from_dataframe(df_content, InfoContent, engine)

    df_user = df_user.astype(object).replace(np.nan, None)
    df_user["first_login_date_TW"] = pd.to_datetime(
        df_user["first_login_date_TW"]
    ).dt.date
    create_table_from_dataframe(df_user, UserProfile, engine)

    chunked_upload_with_validation(
        df_log,
        model_class=LogProblem,
        engine=engine,
        table_name="log_problem",
    )
