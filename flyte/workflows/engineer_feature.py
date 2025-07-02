from flytekit import workflow
from flytekit.types.structured import StructuredDataset

from flyte.tasks.engineer_feature import create_upid_acc_task


@workflow
def feature_engineering_wf(
    df_log: StructuredDataset,
    # df_user: StructuredDataset,
    # df_content: StructuredDataset
) -> str:
    """Flyte workflow to perform feature engineering on preprocessed data.

    Args:
        df_log (StructuredDataset): Preprocessed log DataFrame.
        df_user (StructuredDataset): Preprocessed user DataFrame.
        df_content (StructuredDataset): Preprocessed content DataFrame.

    Returns:
        str: Message indicating successful completion of feature engineering.
    """
    df_log = create_upid_acc_task(df_log=df_log)
    # m_concept_proficiency = create_concept_proficiency_task(df_log=df_log, df_content=df_content)
    # m_proficiency = create_level4_proficiency_task(df_log=df_log, df_user=df_user, df_content=df_content)
    return "Feature engineering completed successfully."
