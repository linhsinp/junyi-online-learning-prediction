from flytekit import workflow

from flyte.workflows.engineer_feature import feature_engineering_wf
from flyte.workflows.preprocess import preprocessing_wf

# from flyte.workflows.train_model import training_wf


@workflow
def full_pipeline_wf() -> str:
    """
    Full ML pipeline: raw ingestion → preprocessing → feature engineering → training
    Each step calls its own sub-workflow.
    """
    df_log, df_user, df_content = preprocessing_wf()
    feature_engineering_wf(df_log, df_user, df_content)
    return "Preprocessing and feature engineering completed successfully."
