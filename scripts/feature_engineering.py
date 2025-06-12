"""
PROBLEM-LEVEL PREDICTION Feature Engineering
The aim is to predict whether a student will answer a problem correct given the details of the problem and the student's performance history. 
"""

import numpy as np 
import pandas as pd 
import os


# Path
PATH_OUTPUT = 'data/output'
PATH_PREPROCESSED_INPUT = 'data/output'

# Files
FILE_LOG_PROCESSED = os.path.join(PATH_PREPROCESSED_INPUT ,'Processed_Log_Problem_raw_timestamp.parquet.gzip')
FILE_USER_PROCESSED = os.path.join(PATH_PREPROCESSED_INPUT ,'Processed_Info_UserData.parquet.gzip')
FILE_CONTENT_PROCESSED = os.path.join(PATH_PREPROCESSED_INPUT ,'Processed_Info_Content.parquet.gzip')


def create_upid_acc(df_log: pd.DataFrame) -> pd.DataFrame:
    """
    Create UPID accuracy features in the log DataFrame.

    This function computes the running accuracy for each UPID and for each user-UPID pair,
    using only past data. It fills missing values with the grand average accuracy.
    
    Args:
        df_log (pd.DataFrame): The log DataFrame containing 'upid' and 'is_correct' columns.
    Returns:
        pd.DataFrame: The log DataFrame with added columns for UPID accuracy and user-UPID accuracy.
    
    """
    ACC_GRAND_AVG = df_log.is_correct.mean()        
    # Compute running accuracy for each upid, using only past data 
    #  - for problem-level features
    df_log['v_upid_acc'] = (
        df_log
        .groupby('upid', observed=True)['is_correct']
        .expanding()
        .mean()
        .shift(1)
        .fillna(ACC_GRAND_AVG)
        .reset_index(level=0, drop=True)
    )
    # Compute running accuracy for each user and upid, using only past data 
    #  - for personalized prediction
    df_log['v_uuid_upid_acc'] = (
        df_log
        .groupby(['uuid', 'upid'], observed=True)['is_correct']
        .expanding()
        .mean()
        .shift(1)
        .fillna(ACC_GRAND_AVG)
        .reset_index(level=[0,1], drop=True)
    )
    return df_log


def create_concept_proficiency(df_log: pd.DataFrame, list_concept_id: np.array) -> pd.DataFrame:
    """
    Create concept proficiency features in the log DataFrame.

    This function computes the most recent level for each concept before each log entry.

    Args:
        df_log (pd.DataFrame): The log DataFrame containing 'ucid' and 'level' columns.
        list_concept_id (np.array): An array of unique concept IDs.

    Returns:
        pd.DataFrame: The log DataFrame with an added column for concept proficiency.
    """
    recent_level = {cid: np.nan for cid in list_concept_id}
    concept_proficiency_list = []
    for row_id, log in df_log.iterrows():
        if row_id % 1000000 == 0:                
            print(row_id)
        cid = log['ucid']
        # Append the most recent level for this concept (before this log)
        concept_proficiency_list.append(recent_level[cid])
        # Update the most recent level for this concept
        recent_level[cid] = log['level']
    # Add the concept proficiency to the DataFrame
    df_log['concept_proficiency'] = concept_proficiency_list
    return df_log


def create_level4_proficiency_matrix(
    df_log: pd.DataFrame, 
    df_content: pd.DataFrame, 
    list_concept_id: np.array, 
    dict_concept_id: dict, 
    list_level4_id: np.array,
    dict_level4_id: dict,
    list_user_id: pd.Categorical,
    dict_user_id: dict
) -> None:
    """
    Create a proficiency matrix for level-4 categories based on user logs and content data. (# logs, # level4 id)

    The student's most recent "level" of a level-4 category, which is derived by 
    averaging across the most recent levels of all concepts within one "level-4" category.

    Args:
        df_log (pd.DataFrame): The log DataFrame containing user interactions.
        df_content (pd.DataFrame): The content DataFrame containing concept and level-4 information.
        list_concept_id (np.array): An array of unique concept IDs.
        dict_concept_id (dict): A dictionary mapping concept IDs to their indices.
        list_level4_id (np.array): An array of unique level-4 IDs.
        dict_level4_id (dict): A dictionary mapping level-4 IDs to their indices.
        list_user_id (pd.Categorical): A list of unique user IDs.
        dict_user_id (dict): A dictionary mapping user IDs to their indices.
    
    Returns:
        None: Saves the proficiency matrix as a compressed numpy file.
    """
    
    # Create a mapping from level-4 id to concept ids
    dict_level4_to_ucid = (
        df_content
        .groupby('level4_id', observed=True)['ucid']
        .apply(lambda ucids: [dict_concept_id[ucid] for ucid in ucids])
        .to_dict()
    )
    # Convert level-4 id to numeric index to map with proficiency matrix
    dict_level4_to_ucid_new = {}
    for key in dict_level4_to_ucid.keys():
        numeric_id = dict_level4_id[key]
        dict_level4_to_ucid_new[numeric_id] = dict_level4_to_ucid[key]

    # Create the "proficiency matrix" (# logs, # level 4 id) which encodes the most recent level per level-4 category (averaged across ucid)
    m_proficiency = np.empty((len(df_log), len(list_level4_id)), dtype = 'float16')
    m_proficiency[:] = np.nan

    # Create the helper "concept level matrix" (# users, # concept id) which encodes the most recent level per concept of each student  
    m_concept_level = np.empty((len(list_user_id), len(list_concept_id))) #, dtype = 'int8'
    m_concept_level[:] = np.nan   

    for row_id, log in df_log.iterrows():
        if row_id % 10000 == 0:                
            print(row_id)
        # Update the "concept level matrix"
        m_concept_level[dict_user_id[log['uuid']], dict_concept_id[log['ucid']]] = log['level']
        # Update the "proficiency matrix" with the average concept level within the level 4 id
        m_proficiency[row_id, dict_level4_id[log['level4_id']]] = (
            np.nansum(m_concept_level[dict_user_id[log['uuid']], dict_level4_to_ucid_new[dict_level4_id[log['level4_id']]]])
        )                             

    # We need to convert nan values present in proficiency matrix to 0. 
    # Otherwise, we will get an exception when training the model.
    m_proficiency[np.isnan(m_proficiency)] = 0

    # save the m_proficiency matrix
    np.savez_compressed(os.path.join(PATH_OUTPUT,'m_proficiency_level4'), m_proficiency)


def load_proficiency_matrix():
    """
    Load the proficiency matrix from the saved file and perform checks.

    This function checks if the matrix is not empty and prints selected values where they are not NaN or 0.
    """
    m_proficiency = np.load(os.path.join(PATH_OUTPUT,'m_proficiency_level4.npz'))['arr_0']
    print("Proficiency matrix created and saved successfully.")
    # Check if the matrix is not empty
    if m_proficiency.size == 0:
        raise ValueError("Proficiency matrix is empty. Please check the input data.")
    # Check the selected values in m_proficiency where values are not NaN
    print("Selected values in m_proficiency where values are not NaN:")
    for i in range(len(m_proficiency)):     
        if np.any(~np.isnan(m_proficiency[i, :])):
            print(f"Row {i}: {m_proficiency[i, ~np.isnan(m_proficiency[i, :])][:10]}")
    # Check the selected values in m_proficiency where values are not 0
    print("Selected values in m_proficiency where values are not 0:")
    for i in range(len(m_proficiency)):
        if np.any(m_proficiency[i, :] != 0):
            print(f"Row {i}: {m_proficiency[i, m_proficiency[i, :] != 0][:10]}")


if __name__ == "__main__":

    # Load the preprocessed data
    if not os.path.exists(FILE_LOG_PROCESSED):
        raise FileNotFoundError(f"Log file not found: {FILE_LOG_PROCESSED}")
    if not os.path.exists(FILE_USER_PROCESSED):
        raise FileNotFoundError(f"User file not found: {FILE_USER_PROCESSED}")
    if not os.path.exists(FILE_CONTENT_PROCESSED):
        raise FileNotFoundError(f"Content file not found: {FILE_CONTENT_PROCESSED}")
    df_user = pd.read_parquet(FILE_USER_PROCESSED)
    df_content = pd.read_parquet(FILE_CONTENT_PROCESSED)
    df_log = pd.read_parquet(FILE_LOG_PROCESSED)
    df_log = df_log.sort_values(['timestamp_TW', 'uuid', 'upid']).reset_index(drop=True)

    # Initialize variables
    list_concept_id = df_content.ucid.to_numpy()
    dict_concept_id = {id: order for order, id in enumerate(list_concept_id)}
    list_level4_id = df_content.level4_id.unique().to_numpy()
    dict_level4_id = {id:order for order, id in enumerate(list_level4_id)}
    list_user_id = df_user['uuid'].unique()
    dict_user_id = {id:order for order, id in enumerate(list_user_id)}

    # Save the updated log DataFrame with UPID accuracy and concept proficiency
    df_log = create_upid_acc(df_log)

    # Create concept proficiency features
    df_log = create_concept_proficiency(df_log, list_concept_id)

    # Join the level 4 info to the log DataFrame
    required_columns = {"ucid", "level4_id"}
    if not required_columns.issubset(df_content.columns):
        raise ValueError(f"df_content is missing required columns: {required_columns - set(df_content.columns)}")
    df_log = df_log.merge(df_content[["ucid", "level4_id"]], how="left")

    create_level4_proficiency_matrix(
        df_log, 
        df_content, 
        list_concept_id, 
        dict_concept_id, 
        list_level4_id, 
        dict_level4_id, 
        list_user_id, 
        dict_user_id
    )

    print("Feature engineering completed and saved successfully.")