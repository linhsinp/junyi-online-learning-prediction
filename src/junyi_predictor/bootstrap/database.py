"""Helpers for seeding PostgreSQL from local raw artifacts."""

from __future__ import annotations

import logging
import math
from collections import Counter
from datetime import date
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd
from pydantic import ValidationError
from sqlalchemy import text
from sqlalchemy.engine import Engine
from sqlmodel import Field, Session, SQLModel, create_engine

from junyi_predictor.paths import CONTENT_FILE, USER_FILE
from junyi_predictor.progress import timed

logger = logging.getLogger(__name__)

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


ENUM_MAPS = {
    "info_content": ENUM_MAP_CONTENT,
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


@timed("bootstrap.validate")
def validate_with_sqlmodel(
    df: pd.DataFrame,
    model_class: type[SQLModel],
    enum_maps: dict[str, dict[str, type[Enum]]] = ENUM_MAPS,
) -> pd.DataFrame:
    """Validate DataFrame rows against a SQLModel schema with optional Enum coercion."""
    table_name = getattr(model_class, "__tablename__", None)
    enum_map = enum_maps.get(table_name)

    valid_data: list[dict[str, Any]] = []
    rejected_rows = 0
    reasons: Counter[str] = Counter()
    for _, row in df.iterrows():
        try:
            row_dict = row.to_dict()
            if enum_map:
                row_dict = to_enum_aware_dict(row_dict, enum_map)
            record = model_class(**row_dict)
            valid_data.append(record.model_dump())
        except ValidationError as error:
            rejected_rows += 1
            reasons.update(
                item["type"]
                for item in error.errors(include_input=False, include_context=False)
            )
    if rejected_rows:
        logger.warning(
            "Rows rejected during validation",
            extra={
                "event": "validation.rejected",
                "table": table_name,
                "rejected_rows": rejected_rows,
                "accepted_rows": len(valid_data),
                "reasons": dict(reasons),
            },
        )
    return pd.DataFrame(valid_data)


@timed("bootstrap.create_table")
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


@timed("bootstrap.upload_chunks", fields=("table_name",))
def chunked_upload_with_validation(
    df: pd.DataFrame,
    model_class: type[SQLModel],
    engine: Engine,
    table_name: str,
    chunk_size: int = 1_000,
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
    logger.info(
        "Chunks uploaded",
        extra={
            "event": "database.chunks_uploaded",
            "chunk_count": total_chunks,
            "table": table_name,
        },
    )


@timed("bootstrap.reset_database")
def reset_database(engine_url: str = DEFAULT_ENGINE_URL) -> None:
    """Drop source and workflow-output tables so the database can be rebuilt."""
    engine = create_engine(engine_url)
    with engine.begin() as connection:
        connection.execute(text("DROP TABLE IF EXISTS log_problem CASCADE"))
        connection.execute(text("DROP TABLE IF EXISTS processed_log CASCADE"))
        connection.execute(text("DROP TABLE IF EXISTS feature_snapshot CASCADE"))
    SQLModel.metadata.drop_all(engine)


@timed("bootstrap.seed_database")
def seed_database_from_raw_files(engine_url: str = DEFAULT_ENGINE_URL) -> None:
    """Load raw local artifacts and seed the PostgreSQL tables used by the workflows."""
    engine = create_engine(engine_url)
    df_user = pd.read_csv(
        USER_FILE,
        dtype={"uuid": "category", "gender": "category", "user_grade": "int8"},
    )
    df_content = pd.read_csv(
        CONTENT_FILE,
        dtype={
            "ucid": "category",
            "level4_id": "category",
            "difficulty": "category",
            "learning_stage": "category",
        },
    )

    create_table_from_dataframe(df_content, InfoContent, engine)

    df_user = df_user.astype(object).replace(np.nan, None)
    df_user["first_login_date_TW"] = pd.to_datetime(
        df_user["first_login_date_TW"]
    ).dt.date
    create_table_from_dataframe(df_user, UserProfile, engine)
