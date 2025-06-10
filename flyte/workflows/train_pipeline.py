
from flytekit import workflow
from flyte.tasks.ingest import ingest_data_gcs
from flyte.tasks.train import train_model

@workflow
def full_pipeline():
    data_path = ingest_data_gcs(bucket="my-bucket", blob="raw/junyi.csv")
    model_path = train_model(data_path=data_path)
    return model_path
