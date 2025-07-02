import pandas as pd
from flytekit import StructuredDataset, task

from scripts.engineer_feature import (
    create_concept_proficiency,
    create_level4_proficiency_matrix,
    create_upid_acc,
)

custom_image = "linhsinp/junyi-predictor-image:latest"


@task(container_image=custom_image)
def create_upid_acc_task(df_log: StructuredDataset) -> StructuredDataset:
    """Flyte task to create UPID accuracy features in the log DataFrame.

    This function adds UPID accuracy features to the log DataFrame by calculating
    the accuracy of user performance on each concept.

    Args:
        df_log (StructuredDataset): Preprocessed log DataFrame.

    Returns:
        StructuredDataset: Log DataFrame with UPID accuracy features added.
    """
    print("Starting UPID accuracy feature engineering...")
    df_log = df_log.open(pd.DataFrame).all()
    df_log = create_upid_acc(df_log)
    print("UPID accuracy feature engineering completed.")
    return StructuredDataset(df_log)


@task(container_image=custom_image)
def create_concept_proficiency_task(
    df_log: StructuredDataset, df_content: StructuredDataset
) -> StructuredDataset:
    """Flyte task to create concept proficiency matrix from log and content DataFrames.

    This function computes the concept proficiency matrix based on user logs and content data.

    Args:
        df_log (StructuredDataset): Preprocessed log DataFrame.
        df_content (StructuredDataset): Preprocessed content DataFrame containing concept IDs.

    Returns:
        StructuredDataset: Concept proficiency matrix.
    """
    df_log = df_log.open(pd.DataFrame).all()
    df_content = df_content.open(pd.DataFrame).all()
    print("Starting concept proficiency feature engineering...")
    list_concept_id = df_content.ucid.to_numpy()
    dict_concept_id = {id: order for order, id in enumerate(list_concept_id)}
    m_concept_proficiency = create_concept_proficiency(
        df_log, list_concept_id, dict_concept_id
    )
    print("Concept proficiency feature engineering completed.")
    return StructuredDataset(m_concept_proficiency)


@task(container_image=custom_image)
def create_level4_proficiency_task(
    df_log: StructuredDataset, df_user: StructuredDataset, df_content: StructuredDataset
) -> StructuredDataset:
    """Flyte task to create level-4 proficiency matrix from log, user, and content DataFrames.

    This function computes the level-4 proficiency matrix based on user logs, user data, and content data.

    Args:
        df_log (StructuredDataset): Preprocessed log DataFrame.
        df_user (StructuredDataset): Preprocessed user DataFrame containing user IDs.
        df_content (StructuredDataset): Preprocessed content DataFrame containing concept IDs and level-4 IDs.

    Returns:
        StructuredDataset: Level-4 proficiency matrix.
    """
    print("Starting level-4 proficiency feature engineering...")
    list_concept_id = df_content.ucid.to_numpy()
    dict_concept_id = {id: order for order, id in enumerate(list_concept_id)}
    list_level4_id = df_content.level4_id.unique().to_numpy()
    dict_level4_id = {id: order for order, id in enumerate(list_level4_id)}
    list_user_id = df_user["uuid"].unique()
    dict_user_id = {id: order for order, id in enumerate(list_user_id)}
    m_proficiency = create_level4_proficiency_matrix(
        df_log,
        df_content,
        list_concept_id,
        dict_concept_id,
        list_level4_id,
        dict_level4_id,
        list_user_id,
        dict_user_id,
    )
    return StructuredDataset(m_proficiency)
