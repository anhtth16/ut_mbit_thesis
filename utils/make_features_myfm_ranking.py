
import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder

'''
SCRIPT FUNCTION: BUILD FEATURE VECTOR FOR TOP-N RECOMMENDATION 
for factorization machine model, using myFM package

ASSUMPTIONS:
- Warm users (already have information in user_set)
- With work history (already have information in work_history)

INPUTS:
- clean job_set, user_set

OUTPUTS:
- X_rec: sparse matrix
- Dimension of X depending on the type of myFM classifier 
'''

'''
BEFORE RUNNING THIS SCRIP, please load following data
# Load clean job data
job_set = pd.read_csv("./data_processed/jobset_clean.csv")
user_set = pd.read_csv("./data_interim/user_set_cleaned.csv")
dataset = pd.read_csv("./data_interim/dataset_cleaned.csv")

# Load discretize data for all users and jobs
job_fm = pd.read_csv('./data_interim/jobs_fm.csv')
user_fm = pd.read_csv('./data_interim/users_fm.csv')

train_user = user_set[user_set.Split=="Train"].UserID.values
test_user = user_set[user_set.Split=="Test"].UserID.values
train_data = dataset[dataset.UserID.isin(train_user)]
test_data = dataset[dataset.UserID.isin(test_user)]
'''

'''
1 - Build groups dataframe with following columns
- UserID: one single test user id that need recommendation
- JobID: (all jobs in the top 15% popular that can be pair with UserID)
- City: get_city_match
- State: get_state_match
- Country: get_country_match
'''

def build_group(test_uid, ranking_data):
    groups = ranking_data[ranking_data.UserID == test_uid]
    return groups

'''
FEATURE_COLUMNS: depends on type of myFM model: fm, fm_match, fm_side_info, or fm_extended
'''
def get_feature_column(fm_type):
    if fm_type in ['fm', 'fm_side_info']: 
        FEATURE_COLUMNS = ['UserID', 'JobID']
    elif fm_type in ['fm_match', 'fm_extended']:
        FEATURE_COLUMNS = ['UserID', 'JobID', 'City', 'State', 'Country']
    return FEATURE_COLUMNS

# Get user and job side information
def get_user_info(test_uid, user_fm):
    user_info = user_fm[user_fm.UserID==test_uid].set_index('UserID')
    return user_info

def get_job_info(test_uid, job_fm, ranking_data):
    jobs_list = ranking_data.JobID.tolist()
    job_info = job_fm[job_fm.JobID.isin(jobs_list)].set_index('JobID')
    return job_info

'''
Start building feature vector:
- For fm, fm_match: build_feature()
- For fm_side_info and fm_extended: build_feature_extended()
'''

def build_feature(fm_type, groups, train_data):
    if fm_type == 'fm':
        FEATURE_COLUMNS = ['UserID', 'JobID']
        ohe = OneHotEncoder(handle_unknown='ignore')
        X_train = ohe.fit_transform(train_data[FEATURE_COLUMNS])
        X_rec = ohe.transform(groups[FEATURE_COLUMNS])
    elif fm_type == 'fm_match':
        FEATURE_COLUMNS = ['UserID', 'JobID', 'City', 'State', 'Country']
        ohe_match = OneHotEncoder(handle_unknown='ignore')
        X_train = ohe_match.fit_transform(train_data[FEATURE_COLUMNS])
        X_rec = ohe_match.transform(groups[FEATURE_COLUMNS])
    return X_rec


def build_feature_extended(fm_type, groups, train_data, user_fm, job_fm, ranking_data, test_uid):
    user_info = get_user_info(test_uid, user_fm)
    job_info = get_job_info(test_uid, job_fm, ranking_data)
    
    user_info_ohe = OneHotEncoder(handle_unknown='ignore').fit(user_fm.set_index('UserID'))
    job_info_ohe = OneHotEncoder(handle_unknown='ignore').fit(job_fm.set_index('JobID'))
    
    if fm_type == 'fm_side_info':
        X_simple = build_feature('fm', groups=groups, train_data = train_data)
        
        # Extend the X_simple (pure interaction) and user, job info
        import scipy.sparse as sps
        X_rec = sps.hstack([
            X_simple,
            user_info_ohe.transform(
                user_info.reindex(groups.UserID)
            ),
            job_info_ohe.transform(
                job_info.reindex(groups.JobID)
            )])
    
    elif fm_type == 'fm_extended':
        X_match = build_feature('fm_match', groups=groups, train_data=train_data)
        
        # Extend the X_match (interaction+match info) and user, job info
        import scipy.sparse as sps
        X_rec = sps.hstack([
            X_match,
            user_info_ohe.transform(
                user_info.reindex(groups.UserID)
            ),
            job_info_ohe.transform(
                job_info.reindex(groups.JobID)
            )])
    return X_rec

'''
Function: get_group_shapes as a parameter when training extended FM models
'''

def get_group_shapes(fm_type, groups): # Number of categories for each encoder
    user_info_ohe = OneHotEncoder(handle_unknown='ignore').fit(user_info)
    job_info_ohe = OneHotEncoder(handle_unknown='ignore').fit(job_info)
    ohe = OneHotEncoder(handle_unknown='ignore').fit(groups[['UserID', 'JobID']])
    ohe_match = OneHotEncoder(handle_unknown='ignore').fit(groups[['UserID', 'JobID', 'City', 'State', 'Country']])
    
    if fm_type == 'fm_match':       
        group_shapes = (
            [len(group) for group in ohe.categories_] + # One-hot encoding user_id, job_id
            [len(group) for group in user_info_ohe.categories_] + # One-hot encoding side information of user 
            [len(group) for group in job_info_ohe.categories_]  # One-hot encoding for job
        )
    elif fm_type == 'fm_extended':
        # Number of categories for each encoder
        group_shapes = (
            [len(group) for group in ohe_match.categories_] + # One-hot encoding user_id, job_id, 3 types of matching
            [len(group) for group in user_info_ohe.categories_] + # One-hot encoding side information of user 
            [len(group) for group in job_info_ohe.categories_]  # One-hot encoding for job
        )
    
    return group_shapes
