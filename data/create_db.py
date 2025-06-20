import math
from datetime import date
from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd
from pydantic import ValidationError
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


# def to_enum_aware_dict(row: dict, enum_map: dict[str, Enum]) -> dict:
#     return {k: enum_map[k](v) if k in enum_map else v for k, v in row.items()}


def to_enum_aware_dict(row: dict, enum_map: dict[str, type[Enum]]) -> dict:
    def safe_enum(enum_cls, value):
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return None
        return enum_cls(value)

    return {
        k: safe_enum(enum_map[k], v)
        if k in enum_map
        else (None if isinstance(v, float) and math.isnan(v) else v)
        for k, v in row.items()
    }


def validate_with_sqlmodel(
    df: pd.DataFrame, model_class: type[SQLModel]
) -> pd.DataFrame:
    enum_maps = {
        "info_content": ENUM_MAP_CONTENT,
        # "user_profile": ENUM_MAP_USER,
    }
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


def create_dbt_from_df(df: pd.DataFrame, model_class: type[SQLModel]) -> None:
    """Creat database table from pandas dataframe.

    Args:
        df (pd.DataFrame): Pandas dataframe from raw csv file.
        model_class (type[SQLModel]): Target data model for validation into database.
    """

    print("Validating data ...")
    validated_df = validate_with_sqlmodel(df, model_class)

    # Create the table if it doesn't exist
    SQLModel.metadata.create_all(sqlmodel_engine)
    print(f"Database table created: {model_class.__tablename__}")

    # Insert validated rows
    with Session(sqlmodel_engine) as session:
        objects = [model_class(**row) for row in validated_df.to_dict(orient="records")]
        session.add_all(objects)
        session.commit()
        print("Data uploaded to PostgreSQL.")


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
    # print(type(df_user['first_login_date_TW'].iloc[0])) # <class 'datetime.date'>
    create_dbt_from_df(df_user, UserProfile)

    # Read first rows the table for checking
    df_read = pd.read_sql("SELECT * FROM user_profile LIMIT 10;", sqlmodel_engine)
    print(df_read)

    # Delete/Drop a table
    from sqlalchemy import text

    with sqlmodel_engine.connect() as conn:
        conn.execute(text("DROP TABLE IF EXISTS user_profile"))
        conn.commit()
