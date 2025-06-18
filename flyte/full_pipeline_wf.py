# from typing import Tuple

# from flytekit import workflow

# from flyte.workflows.engineer_feature import feature_engineering_wf
# from flyte.workflows.preprocess import preprocessing_wf
# from flyte.workflows.train_model import training_wf


# @workflow
# def full_pipeline_wf() -> Tuple[str, str, str]:
#     """
#     Full ML pipeline: raw ingestion → preprocessing → feature engineering → training
#     Each step calls its own sub-workflow.
#     """
#     preprocessed = preprocessing_wf()
#     features = feature_engineering_wf()
#     metrics = training_wf()

#     return preprocessed, features, metrics
