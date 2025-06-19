import pandas as pd
from sqlalchemy import create_engine

host = "localhost"
port = 30001
database = "postgres"
user = "postgres"
password = "postgres"


# Example DataFrame
df = pd.DataFrame({"name": ["Alice", "Bob"], "score": [90, 85]})

# Define DB connection string
engine = create_engine("postgresql://postgres:postgres@localhost:30001/postgres")

# Write to a table (replace if exists)
df.to_sql("student_scores", engine, if_exists="replace", index=False)

print("Data uploaded to PostgreSQL.")

# Read the table
df_read = pd.read_sql("SELECT * FROM student_scores", engine)
print(df_read)
