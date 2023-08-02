'''
FUNCTION: Make top_N recommendation for a single user id
Input:
    - model: 4 myfm models 
    - N: number of recommendation 20
    - ranking_data: ranking_data_knn_lda
    - test_user
    - user_fm: dataset. MUST DROP 'Split' Column before running
    - job_fm
Output: rec_results
'''

import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder
from sklearn import metrics
import myfm
from make_features_myfm_ranking import * # Import customer script for building features


def topN_fm_simple(u_id, fm_type, fm_model, ranking_data, train_data, N):
    allowed_values = ["fm", "fm_match"] # fm_type can be 'fm' or 'fm_match'
    if fm_type is None:
        fm_type = allowed_values[0]  # Set the first default value

    if fm_type not in allowed_values:
        raise ValueError("Invalid value for fm_type. Allowed values are: {}".format(allowed_values))

    # Build group
    groups_rec = build_group(test_uid=u_id, ranking_data = ranking_data)
    
    # Build recommendation vector
    X_rec = build_feature(fm_type = fm_type, 
                          groups = groups_rec,
                          train_data = train_data)
    # Make prediction
    Y_pred = fm_model.predict(X_rec).astype('int')
    Y_prob = fm_model.predict_proba(X_rec)

    # Get list 20 jobs
    job_id_list = groups_rec.JobID.values
    rec_N = sorted(
            [
                (ids_j, yprob_j, ypred_j) for yprob_j, ypred_j, ids_j in zip(Y_prob, Y_pred, job_id_list)
            ],
            key=lambda x: -x[1]
        )[0:N]
    return rec_N

def topN_fm_extend(u_id, fm_type, fm_model, ranking_data, train_data, user_fm, job_fm, N):
    allowed_values = ["fm_side_info", "fm_extended"] # fm_type can be 'fm_side_info' or 'fm_extended'
    if fm_type is None:
        fm_type = allowed_values[0]  # Set the first default value

    if fm_type not in allowed_values:
        raise ValueError("Invalid value for fm_type. Allowed values are: {}".format(allowed_values))
    
    # Build group
    groups_rec = build_group(test_uid=u_id, ranking_data = ranking_data)
    
    # Build feature
    X_rec = build_feature_extended(fm_type = fm_type, # Build recommendation vector
                              groups = groups_rec,
                              train_data = train_data,
                                user_fm = user_fm,
                               job_fm = job_fm, 
                               ranking_data = ranking_data, 
                               test_uid = u_id)

    Y_pred = fm_model.predict(X_rec).astype('int') # Make prediction
    Y_prob = fm_model.predict_proba(X_rec)

    # Get list 20 jobs
    job_id_list = groups_rec.JobID.values
    rec_N = sorted(
            [
                (ids_j, yprob_j, ypred_j) for yprob_j, ypred_j, ids_j in zip(Y_prob, Y_pred, job_id_list)
            ],
            key=lambda x: -x[1]
        )[0:N]
    return rec_N

def get_rec_result_df(u_id, rec_N): # Convert recN result to dataframe
    rec_N_df = pd.DataFrame(rec_N)
    rec_cols = ['JobID', 'Y_prob', 'Y_pred']
    rec_N_df.columns = rec_cols
    rec_N_df['UserID'] = u_id
    rec_N_df['rank'] = rec_N_df.groupby('UserID').cumcount()
    rec_N_df = rec_N_df[['UserID','JobID', 'Y_prob', 'Y_pred', 'rank']]
    return rec_N_df
