'''
FUNCTION: Make top_N recommendation: for different models: 
    - topN_ebm_location: model ebm (only location)
    - topN_ebm_side_info: model ebm_side_info, or 
    - topN_ebm__extended: model ebm_extended (full information)
Input:
    - model: baseline binary classification model (sklearn)
    - N: number of recommendation 20
    - groups: ranking_data
    - test_user
Output: rec_results
'''
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm
import random
import warnings
warnings.filterwarnings("ignore")

# Import library for baseline classification models
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score, precision_score, f1_score, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis

# lda jobs features
features_job = ['ReqTopic', 'DescTopic', 'TitTopic']
# lda & user profile features
features_user = ['DegreeType', 'CurrentlyEmployed', 'ManagedOthers', 'WorkHistoryTopic',
       'WorkHistoryLevel', 'SeniorLevel']
# location features
features_location = ['City', 'State', 'Country']

def topN_ebm_extended(model, N, ranking_data, features_df): # features_df should be imported before running this script
    user_ids = list(ranking_data.groupby('UserID').UserID.unique().astype('int'))
    rec_result = {}
#     mark_stop = 0
    for u_id in user_ids:
        # print('UserID:', u_id)
        group = ranking_data[ranking_data.UserID ==u_id]
        job_ids = ranking_data[ranking_data.UserID ==u_id].JobID.values
        rec_items = []
        
        for j_id in job_ids:
    
            # Build feature: location + user profile + work_history_matrix + job_matrix
            location = group[(group.UserID == u_id) & (group.JobID == j_id)].reset_index()[features_location].loc[0]
            # print(location)

            # Get user profile and work history features
            user = features_df[features_df.UserID==u_id].reset_index()[features_user].loc[0]

            # Get job details
            job_details = features_df[features_df.JobID==j_id].reset_index()[features_job].loc[0]
            # Concat all
            test_feature = pd.concat([location, user, job_details], axis = 0)
            test_df = pd.DataFrame(test_feature).T

            # Make prediction on the feature vector, result is probability for each class [0, 1]
            yprob_j = model.predict_proba(test_df)
            ypred_j = model.predict(test_df)
            zip_result = (j_id, float(yprob_j[:,1]), ypred_j[0])
            rec_items.append(zip_result)
            rec_items = sorted(rec_items, key=lambda x: -x[1])
            rec_N = rec_items[:N]
            
            rec_result[u_id] = rec_N
#         mark_stop = mark_stop + 1
       
#         if mark_stop ==3:
#             break
    return rec_result

def topN_ebm_side_info(model, N, ranking_data, features_df): # features_df should be imported before running this script
    user_ids = list(ranking_data.groupby('UserID').UserID.unique().astype('int'))
    rec_result = {}
#     mark_stop = 0
    for u_id in user_ids:
        # print('UserID:', u_id)
        group = ranking_data[ranking_data.UserID ==u_id]
        job_ids = ranking_data[ranking_data.UserID ==u_id].JobID.values
        rec_items = []
        
        for j_id in job_ids:
            # Get user profile and work history features
            user = features_df[features_df.UserID==u_id].reset_index()[features_user].loc[0]

            # Get job details
            job_details = features_df[features_df.JobID==j_id].reset_index()[features_job].loc[0]
            # Concat all
            test_feature = pd.concat([user, job_details], axis = 0)
            test_df = pd.DataFrame(test_feature).T

            # Make prediction on the feature vector, result is probability for each class [0, 1]
            yprob_j = model.predict_proba(test_df)
            ypred_j = model.predict(test_df)
            zip_result = (j_id, float(yprob_j[:,1]), ypred_j[0])
            rec_items.append(zip_result)
            rec_items = sorted(rec_items, key=lambda x: -x[1])
            rec_N = rec_items[:N]
            
            rec_result[u_id] = rec_N
#         mark_stop = mark_stop + 1
       
#         if mark_stop ==3:
#             break
    return rec_result

def topN_ebm_location(model, N, ranking_data, features_df): # features_df should be imported before running this script
    user_ids = list(ranking_data.groupby('UserID').UserID.unique().astype('int'))
    rec_result = {}
#     mark_stop = 0
    for u_id in user_ids:
        # print('UserID:', u_id)
        group = ranking_data[ranking_data.UserID ==u_id]
        job_ids = ranking_data[ranking_data.UserID ==u_id].JobID.values
        rec_items = []
        
        for j_id in job_ids:
            # Build feature: location + user profile + work_history_matrix + job_matrix
            location = group[(group.UserID == u_id) & (group.JobID == j_id)].reset_index()[features_location].loc[0]
            test_df = pd.DataFrame(location).T

            # Make prediction on the feature vector, result is probability for each class [0, 1]
            yprob_j = model.predict_proba(test_df)
            ypred_j = model.predict(test_df)
            zip_result = (j_id, float(yprob_j[:,1]), ypred_j[0])
            rec_items.append(zip_result)
            rec_items = sorted(rec_items, key=lambda x: -x[1])
            rec_N = rec_items[:N]
            
            rec_result[u_id] = rec_N
#         mark_stop = mark_stop + 1
       
#         if mark_stop ==3:
#             break
    return rec_result
