import math
from datetime import date, datetime
from enum import Enum
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd
from pydantic import ValidationError
from sqlalchemy.engine import Engine
from sqlmodel import Field, Session, SQLModel, create_engine

from scripts.preprocess import FILE_CONTENT, FILE_LOG_FULL, FILE_USER, load_data_into_df

# Define DB connection string
ENGINE_URL = "postgresql://postgres:postgres@localhost:30001/postgres"

# host = "localhost"
# port = 30001 # sandbox relational database
# database = "postgres" # sandbox default
# user = "postgres" # sandbox default
# password = "postgres" # sandbox default


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


# class GenderEnum(str, Enum):
#     male = "male"
#     female = "female"
#     unspecified = "unspecified"


# ENUM_MAP_USER = {
#     "gender": GenderEnum,
# }


class UserProfile(SQLModel, table=True):
    __tablename__ = "user_profile"
    __table_args__ = {"extend_existing": True}

    uuid: str = Field(primary_key=True)
    gender: Optional[str] = Field(default=None)
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

    timestamp_TW: Optional[datetime] = Field(primary_key=True)
    uuid: Optional[str] = Field(primary_key=True)
    ucid: Optional[str] = Field(foreign_key="info_content.ucid")
    upid: Optional[str] = Field(primary_key=True)

    problem_number: Optional[int] = Field(default=None)
    exercise_problem_repeat_session: Optional[int] = Field(default=None)

    is_correct: Optional[bool] = Field(default=None)
    total_sec_taken: Optional[int] = Field(default=None)
    total_attempt_cnt: Optional[int] = Field(default=None)
    used_hint_cnt: Optional[int] = Field(default=None)
    is_hint_used: Optional[bool] = Field(default=None)

    # These are object dtype, likely stringified booleans ('true', 'false', or NaN)
    is_downgrade: Optional[BoolObjectEnum] = Field(default=None)
    is_upgrade: Optional[BoolObjectEnum] = Field(default=None)

    level: Optional[int] = Field(default=None)


ENUM_MAPS = {
    "info_content": ENUM_MAP_CONTENT,
    # "user_profile": ENUM_MAP_USER,
    "log_problem": ENUM_MAP_LOG_PROBLEM,
}


def to_enum_aware_dict(row: dict, enum_map: dict[str, type[Enum]]) -> dict[str, Any]:
    def safe_enum(enum_cls: type[Enum], value: Any) -> Enum | None:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return None
        try:
            return enum_cls(value)
        except ValueError:
            return None

    return {
        k: safe_enum(enum_map[k], v)
        if k in enum_map
        else (None if isinstance(v, float) and math.isnan(v) else v)
        for k, v in row.items()
    }


def validate_with_sqlmodel(
    df: pd.DataFrame,
    model_class: type[SQLModel],
    enum_maps: dict[str, dict[str, type[Enum]]] = ENUM_MAPS,
) -> pd.DataFrame:
    """
    Validate a DataFrame against a SQLModel schema.
    Automatically applies Enum conversion based on table name.
    """
    table_name = getattr(model_class, "__tablename__", None)
    enum_map = enum_maps.get(table_name)

    valid_data = []
    for i, row in df.iterrows():
        try:
            row_dict = row.to_dict()
            if enum_map:
                row_dict = to_enum_aware_dict(row_dict, enum_map)
            record = model_class(**row_dict)
            valid_data.append(record.model_dump())
        except ValidationError as e:
            print(f"Validation failed for row {i}: {e}")
    return pd.DataFrame(valid_data)


def create_dbt_from_df(
    df: pd.DataFrame,
    model_class: type[SQLModel],
    enum_maps: dict[str, dict[str, type[Enum]]],
) -> None:
    """Creat database table from pandas dataframe.

    Args:
        df (pd.DataFrame): Pandas dataframe from raw csv file.
        model_class (type[SQLModel]): Target data model for validation into database.
    """

    print("Validating data ...")
    validated_df = validate_with_sqlmodel(df, model_class, enum_maps)

    # Create the table if it doesn't exist
    SQLModel.metadata.create_all(sqlmodel_engine)
    print(f"Database table created: {model_class.__tablename__}")

    # Insert validated rows
    with Session(sqlmodel_engine) as session:
        objects = [model_class(**row) for row in validated_df.to_dict(orient="records")]
        session.add_all(objects)
        session.commit()
        print("Data uploaded to PostgreSQL.")


def chunked_upload_with_validation(
    df: pd.DataFrame,
    model_class: type[SQLModel],
    engine: Engine,
    table_name: str,
    chunk_size: int = 100_000,
    enum_maps: Optional[dict[str, Callable]] = ENUM_MAPS,
):
    """
    Validate and upload a large DataFrame in chunks using SQLModel class schema.

    * NOTE: This is an alternative to the create_dbt_from_df function for large datasets.

    Args:
        df (pd.DataFrame): DataFrame to upload.
        model_class (type[SQLModel]): SQLModel class for validation.
        engine (Engine): SQLAlchemy engine for database connection.
        table_name (str): Name of the target database table.
        chunk_size (int): Number of rows per chunk.
        enum_map (Optional[dict[str, Callable]]): Optional mapping for Enum fields.
    """
    total_chunks = math.ceil(len(df) / chunk_size)
    print(f"Starting upload in {total_chunks} chunks...")

    for i in range(0, len(df), chunk_size):
        chunk = df.iloc[i : i + chunk_size]
        print(f"Validating rows {i} to {min(i + chunk_size, len(df))}...")
        validated_chunk = validate_with_sqlmodel(chunk, model_class, enum_maps)

        print(
            f"Uploading chunk {i // chunk_size + 1}/{total_chunks} with {len(validated_chunk)} valid rows..."
        )
        validated_chunk.to_sql(
            table_name, engine, if_exists="append", index=False, method="multi"
        )
    print("All chunks uploaded successfully.")


if __name__ == "__main__":
    sqlmodel_engine = create_engine(ENGINE_URL)

    # Load raw csv files
    df_log, df_user, df_content = load_data_into_df(
        FILE_LOG_FULL, FILE_USER, FILE_CONTENT
    )

    # Create info_content table
    create_dbt_from_df(df_content, InfoContent)

    # Create user_profile table
    df_user = df_user.astype(object).replace(np.nan, None)
    df_user["first_login_date_TW"] = pd.to_datetime(
        df_user["first_login_date_TW"]
    ).dt.date
    create_dbt_from_df(df_user, UserProfile)

    # create log_problem table
    df_log["timestamp_TW"] = pd.to_datetime(df_log["timestamp_TW"], errors="coerce")
    df_log = df_log.where(pd.notnull(df_log), None)
    chunked_upload_with_validation(
        df=df_log,
        model_class=LogProblem,
        engine=sqlmodel_engine,
        table_name="log_problem",
        chunk_size=100_000,
    )

    # # Read first rows the table for checking
    # df_read1 = pd.read_sql("SELECT * FROM info_content LIMIT 10;", sqlmodel_engine)
    # df_read2 = pd.read_sql("SELECT * FROM user_profile LIMIT 10;", sqlmodel_engine)
    # df_read3 = pd.read_sql("SELECT * FROM log_problem LIMIT 10;", sqlmodel_engine)
    # print(df_read1)
    # print(df_read2)
    # print(df_read3)

    # # Delete/Drop a table
    # from sqlalchemy import text
    # with sqlmodel_engine.connect() as conn:
    #     conn.execute(text("DROP TABLE IF EXISTS log_problem;"))
    #     conn.commit()
