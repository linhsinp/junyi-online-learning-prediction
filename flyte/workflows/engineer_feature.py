# from flytekit import workflow
# from flyte.tasks.engineer_feature import generate_features
# from flyte.tasks.ingest import fetch_from_gcs


# @workflow
# def feature_engineering_wf() -> str:
#     preprocessed_path = fetch_from_gcs(prefix="experiment")
#     features_path = generate_features(preprocessed_path=preprocessed_path)
#     return features_path
