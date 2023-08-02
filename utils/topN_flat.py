'''
FUNCTION: Make top_N recommendation
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

# Get feature names list 
location_features = ['City', 'State', 'Country'] # for matching location
user_profile_features = ["DegreeType", "WorkHistoryCount", "TotalYearsExperience", "CurrentlyEmployed", 
                                                "ManagedOthers", "ManagedHowMany"] # for user profile

# Get work_matrix_features
work_matrix_idx = range(50)
features_work_matrix = list(map(lambda x: 'work_matrix_' + str(x), work_matrix_idx))

# Get job_matrix_features
job_matrix_idx = range(100)
features_job_matrix = list(map(lambda x: 'job_matrix_' + str(x), job_matrix_idx)) 

def topN_flat(model, N, ranking_data, features_flat): # features_flat should be imported before running this script
    user_ids = list(ranking_data.groupby('UserID').UserID.unique().astype('int'))
    rec_result = {}
    for u_id in user_ids:
        # print('UserID:', u_id)
        group = ranking_data[ranking_data.UserID ==u_id]
        job_ids = ranking_data[ranking_data.UserID ==u_id].JobID.values
        rec_items = []

        for j_id in job_ids:
            
            # Build feature: location + user profile + work_history_matrix + job_matrix
            location = group[(group.UserID == u_id) & (group.JobID == j_id)].reset_index()[location_features].loc[0]

            # Get user work history features
            user = features_flat[features_flat.UserID==u_id].reset_index()[user_profile_features].loc[0] # Get_user_profile
            work_history = features_flat[features_flat.UserID==u_id].reset_index()[features_work_matrix].loc[0]

            # Get job details
            job_details = features_flat[features_flat.JobID==j_id].reset_index()[features_job_matrix].loc[0]
            
            # Concat all
            test_feature = pd.concat([location, user, work_history, job_details], axis = 0)
            test_df = pd.DataFrame(test_feature).T

            # Make prediction on the feature vector, result is probability for each class [0, 1]
            yprob_j = model.predict_proba(test_df)
            ypred_j = model.predict(test_df)
            zip_result = (j_id, float(yprob_j[:,1]), ypred_j[0])
            rec_items.append(zip_result)
            rec_items = sorted(rec_items, key=lambda x: -x[1])
            rec_N = rec_items[:N]
            
            rec_result[u_id] = rec_N
    return rec_result
