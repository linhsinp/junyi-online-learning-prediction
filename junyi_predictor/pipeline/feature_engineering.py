"""Feature engineering stage contracts and transformations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FeatureStageOutput:
    """Explicit output contract for the feature engineering stage."""

    log: pd.DataFrame
    concept_proficiency: np.ndarray
    level4_proficiency: np.ndarray


def create_upid_accuracy_features(df_log: pd.DataFrame) -> pd.DataFrame:
    """Add running UPID accuracy features that only depend on past observations."""
    df_log = df_log.copy()
    correctness = df_log["is_correct"].astype(float)
    grand_average = correctness.mean()

    upid_groups = correctness.groupby(df_log["upid"], observed=True)
    upid_prior_correct = upid_groups.cumsum() - correctness
    upid_prior_count = upid_groups.cumcount()
    df_log["v_upid_acc"] = (
        upid_prior_correct / upid_prior_count.replace(0, np.nan)
    ).fillna(grand_average)

    user_upid_groups = correctness.groupby(
        [df_log["uuid"], df_log["upid"]], observed=True
    )
    user_upid_prior_correct = user_upid_groups.cumsum() - correctness
    user_upid_prior_count = user_upid_groups.cumcount()
    df_log["v_uuid_upid_acc"] = (
        user_upid_prior_correct / user_upid_prior_count.replace(0, np.nan)
    ).fillna(grand_average)
    return df_log


def create_concept_proficiency_matrix(
    df_log: pd.DataFrame, list_concept_id: np.ndarray, dict_concept_id: dict
) -> np.ndarray:
    """Create a log-by-concept matrix containing the latest observed level for each concept."""
    matrix = np.empty((len(df_log), len(list_concept_id)), dtype="float16")
    matrix[:] = np.nan
    for row_id, log in df_log.iterrows():
        matrix[row_id, dict_concept_id[log["ucid"]]] = log["level"]
    matrix[np.isnan(matrix)] = 0
    return matrix


def create_level4_proficiency_matrix(
    df_log: pd.DataFrame,
    df_content: pd.DataFrame,
    list_concept_id: np.ndarray,
    dict_concept_id: dict,
    list_level4_id: np.ndarray,
    dict_level4_id: dict,
    list_user_id: np.ndarray,
    dict_user_id: dict,
) -> np.ndarray:
    """Create a log-by-level4 matrix containing the user's latest accumulated proficiency."""
    required_columns = {"ucid", "level4_id"}
    if not required_columns.issubset(df_content.columns):
        missing_columns = required_columns - set(df_content.columns)
        raise ValueError(f"df_content is missing required columns: {missing_columns}")

    df_log = df_log.copy().merge(df_content[["ucid", "level4_id"]], how="left")
    dict_level4_to_ucid = (
        df_content.groupby("level4_id", observed=True)["ucid"]
        .apply(lambda ucids: [dict_concept_id[ucid] for ucid in ucids])
        .to_dict()
    )
    dict_level4_to_indices = {
        dict_level4_id[key]: value for key, value in dict_level4_to_ucid.items()
    }

    proficiency = np.empty((len(df_log), len(list_level4_id)), dtype="float16")
    proficiency[:] = np.nan
    concept_level = np.empty((len(list_user_id), len(list_concept_id)))
    concept_level[:] = np.nan

    for row_id, log in df_log.iterrows():
        user_index = dict_user_id[log["uuid"]]
        concept_index = dict_concept_id[log["ucid"]]
        level4_index = dict_level4_id[log["level4_id"]]
        concept_level[user_index, concept_index] = log["level"]
        proficiency[row_id, level4_index] = np.nansum(
            concept_level[user_index, dict_level4_to_indices[level4_index]]
        )

    proficiency[np.isnan(proficiency)] = 0
    return proficiency


def build_feature_stage(
    df_log: pd.DataFrame, df_user: pd.DataFrame, df_content: pd.DataFrame
) -> FeatureStageOutput:
    """Run feature engineering with explicit stage boundaries."""
    featured_log = create_upid_accuracy_features(df_log=df_log)
    list_concept_id = df_content["ucid"].to_numpy()
    dict_concept_id = {
        concept_id: order for order, concept_id in enumerate(list_concept_id)
    }
    list_level4_id = df_content["level4_id"].unique()
    dict_level4_id = {
        level4_id: order for order, level4_id in enumerate(list_level4_id)
    }
    list_user_id = df_user["uuid"].unique()
    dict_user_id = {user_id: order for order, user_id in enumerate(list_user_id)}

    concept_proficiency = create_concept_proficiency_matrix(
        df_log=featured_log,
        list_concept_id=list_concept_id,
        dict_concept_id=dict_concept_id,
    )
    level4_proficiency = create_level4_proficiency_matrix(
        df_log=featured_log,
        df_content=df_content,
        list_concept_id=list_concept_id,
        dict_concept_id=dict_concept_id,
        list_level4_id=list_level4_id,
        dict_level4_id=dict_level4_id,
        list_user_id=list_user_id,
        dict_user_id=dict_user_id,
    )
    return FeatureStageOutput(
        log=featured_log,
        concept_proficiency=concept_proficiency,
        level4_proficiency=level4_proficiency,
    )
