from enum import Enum

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


# SQLModel table definition
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


def to_enum_aware_dict(row: dict, enum_map: dict[str, Enum]) -> dict:
    return {k: enum_map[k](v) if k in enum_map else v for k, v in row.items()}


def validate_with_sqlmodel(
    df: pd.DataFrame, model_class: type[SQLModel]
) -> pd.DataFrame:
    valid_data = []
    for i, row in df.iterrows():
        try:
            if model_class.__tablename__ == "info_content":
                record = model_class(
                    **to_enum_aware_dict(row.to_dict(), ENUM_MAP_CONTENT)
                )
            else:
                record = model_class(**row.to_dict())
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

    # Create database table
    create_dbt_from_df(df_content, InfoContent)

    # Read the table
    df_read = pd.read_sql("SELECT * FROM info_content", sqlmodel_engine)
    print(df_read)

    # Delete/Drop a table
    from sqlalchemy import text

    with sqlmodel_engine.connect() as conn:
        conn.execute(text("DROP TABLE IF EXISTS info_content"))
        conn.commit()
