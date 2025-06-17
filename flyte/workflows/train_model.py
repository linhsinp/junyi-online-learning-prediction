from flytekit import workflow
from tasks.ingest import fetch_from_gcs
from tasks.train_model import train_model


@workflow
def training_wf() -> str:
    features_path = fetch_from_gcs(prefix="feature_store")
    metrics = train_model(features_path=features_path)
    return metrics
