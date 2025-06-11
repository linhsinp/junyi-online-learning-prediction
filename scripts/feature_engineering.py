"""
PROBLEM-LEVEL PREDICTION Feature Engineering
The aim is to predict whether a student will answer a problem correct given the details of the problem and the student's performance history. 
"""

import numpy as np 
import pandas as pd 
import os
from collections import Counter, defaultdict


VARS_LOG_CATEGORY = ['uuid', 'ucid', 'upid']
VARS_CONTENT_CATEGORY = ['level3_id','level4_id']

# Path
PATH_OUTPUT = 'data/output'
PATH_PREPROCESSED_INPUT = 'data/output'

# Files
FILE_LOG_PROCESSED = os.path.join(PATH_PREPROCESSED_INPUT ,'Processed_Log_Problem_raw_timestamp.parquet.gzip')
FILE_USER_PROCESSED = os.path.join(PATH_PREPROCESSED_INPUT ,'Processed_Info_UserData.parquet.gzip')
FILE_CONTENT_PROCESSED = os.path.join(PATH_PREPROCESSED_INPUT ,'Processed_Info_Content.parquet.gzip')


# Load the preprocessed data
if not os.path.exists(FILE_LOG_PROCESSED):
    raise FileNotFoundError(f"Log file not found: {FILE_LOG_PROCESSED}")
if not os.path.exists(FILE_USER_PROCESSED):
    raise FileNotFoundError(f"User file not found: {FILE_USER_PROCESSED}")
if not os.path.exists(FILE_CONTENT_PROCESSED):
    raise FileNotFoundError(f"Content file not found: {FILE_CONTENT_PROCESSED}")
df_log = pd.read_parquet(FILE_LOG_PROCESSED)
df_user = pd.read_parquet(FILE_USER_PROCESSED)
df_content = pd.read_parquet(FILE_CONTENT_PROCESSED)


# Sort the log DataFrame by 'uuid', 'upid' and 'timestamp_TW'
df_log = df_log.sort_values(['timestamp_TW', 'uuid', 'upid']).reset_index(drop=True)


#### Creation of UPID Accuracy: The unique ID of the problem. (n = 25785)
ACC_GRAND_AVG = df_log.is_correct.mean()        

# Compute running accuracy for each upid, using only past data - for problem-level features
df_log['v_upid_acc'] = (
    df_log
    .groupby('upid', observed=True)['is_correct']
    .expanding()
    .mean()
    .shift(1)
    .fillna(ACC_GRAND_AVG)
    .reset_index(level=0, drop=True)
)

# Compute running accuracy for each user and upid, using only past data - for personalized prediction
df_log['v_uuid_upid_acc'] = (
    df_log
    .groupby(['uuid', 'upid'], observed=True)['is_correct']
    .expanding()
    .mean()
    .shift(1)
    .fillna(ACC_GRAND_AVG)
    .reset_index(level=[0,1], drop=True)
)


#### Creation of Concept proficiency: The unique ID of the concept/content (n = 1326)

# For each log, get the most recent level for each concept up to that point
list_concept_id = df_content.ucid.to_numpy()
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
# Add the results as a new column to df_log
df_log['concept_proficiency'] = concept_proficiency_list


