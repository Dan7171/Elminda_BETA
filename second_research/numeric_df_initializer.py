import array
from heapq import merge
from itertools import groupby
from locale import format_string
from pathlib import Path
from matplotlib import pyplot as plt, test
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt
import pandas as pd
import numpy as np
import datetime as dt
#import seaborn as sns
import warnings
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score
#from pandas.core.common import SettingWithCopyWarning
warnings.simplefilter(action= "ignore", category= UserWarning)
#warnings.simplefilter(action= "ignore", category= SettingWithCopyWarning)
import os
from runArguments import args

def get_c_to_group_by_c_dict(df,c):
    """Given a df and a column namc c, creating a dictionary of grouping key= c values to vals = dfs of c values"""
    df_grouped_by_c = df.groupby([c])
    c_to_c_group = {}
    for name,group in df_grouped_by_c: # map all groups by key= name= subject, val = group = subject's df (for time complexity reasons)
        group.sort_values(by=c)
        c_to_c_group[name] = group
    return c_to_c_group


def get_set_without_items(s:set,items:list):
    """Given a set s and a list of items in it, returns a new set without those items"""
    for item in items:
        if item in s:
            s.remove(item)
    return s

def drop_cols(df,cols_to_drop:set):
    """Given a set 'cols_to_drop' of column names, removing the columns in it from the df and returns it"""
    df = df[[c for c in df.columns if c not in cols_to_drop]] # allowing only columns we weren't marked to remove
    return df

def get_double_visit_subjects(subject_to_subject_group):
    """Returning a set of all the subjects in visit (strings) that have more than one visit of 
    the same kind (2 times or more visit 1 or two times ot more visit 2)"""
    
    s = set()
    for sub in subject_to_subject_group:
        sub_group = subject_to_subject_group[sub]
        visit1_cnt = 0
        visit2_cnt = 0
        for index, row in sub_group.iterrows():
            sub_visit = row['visit']
            if sub_visit == 1:
                visit1_cnt += 1
            if sub_visit == 2:
                visit2_cnt += 1
        if visit1_cnt > 1 or visit2_cnt > 1:
            s.add(sub)
    return s

    
def get_df_without_illegal_subjects(df):
    """Given a df, finding illegal subjects in it and dropping them out from the df (dropping out
    all the rows(visits) belong to these subject"""
    subject_to_subject_group = get_c_to_group_by_c_dict(df,c='subject')  # creates the dictionary of key = subject, val = it's df in visits
    invalid_subjects = get_double_visit_subjects(subject_to_subject_group)
    df['remove'] = df['subject'].apply(lambda subject: subject in invalid_subjects) # 'mark' with remove 'label' (a new column) all the illegal subject
    df = df[df['remove'] != True] # drop all rows who belong to the illegal subjects
    df = df.drop(['remove'],axis = 1) # not the 'remove' column is redundant. remove it
    
    return df
def change_categorial_non_numeric_to_numeric(subjects):
    """ Given a df, changing values of categorial-variables to numeric random coding (ints)"""
    subjects.gender = pd.Categorical(subjects.gender)
    subjects['gender'] = subjects.gender.cat.codes
    subjects.Treatment_group =  pd.Categorical(subjects.Treatment_group)
    subjects['Treatment_group'] = subjects.Treatment_group.cat.codes
    subjects.recordingSystem =  pd.Categorical(subjects.recordingSystem)
    subjects['recordingSystem'] = subjects.recordingSystem.cat.codes
    subjects.site =  pd.Categorical(subjects.site)
    subjects['site'] = subjects.site.cat.codes
    return subjects

def generate_prediction_df(bna_path, clinical_path): 
    """Main function of this module.
    returning the data frame of subjects that we will use for models.
    subjects will return with one entry (row) for each subject, 'clean' from
    non-predictive data (like dates,strings...) so will only contain numerical data for prediction,
    each row is a baseline visit of a subject.""" 
    clinical = pd.read_csv(os.path.abspath(clinical_path))
    #technical step: remove all-Nan rows and all-Nan cols (for some reason they where added when openning clinical as csv):
    clinical.dropna(axis=0, how='all', inplace=True) # for some reason when opening this csv it is adding many Nan rows and cols
    clinical.dropna(axis=1, how='all', inplace=True) # so in these two rows we remove them
    bna = pd.read_csv(bna_path)
    if not args['use_gamma_columns']:
        bna.dropna(axis=1, how='any',inplace=True) # removing all cordance missing values and any other missing values
    print(bna)
    #Step 1 -  merge dfs by the column 'subject':
    visits = clinical.merge(bna, how = 'inner',on = ['subject']) #leaving in the df all pairs with the same subject
    #Step 2 - drop out illegal subjects- subjcets with two visits of the same kind (1,1 or 2,2):
    visits = get_df_without_illegal_subjects(visits)
    #Step 3 - transform visits to a shape of one row for one subject:
    subjets_baseline = visits[visits['visit'] == 1] # keep only the first visit for each subject. from now on, each row = different subject
    #Step 4 - categorial non-numeirc to categorial numeric:
    subjects_baseline= change_categorial_non_numeric_to_numeric(subjets_baseline)
    #Step 5 - remove non-predictive features (columns) we don't want to include the prediction (like Dates,ids,visit number and so on...)
    clinical_col_set = set(list(clinical.columns.values))
    

    #==========Important! don't forget to remove subject from allowed cols clinical 5.12===========
    if(args["debug"]):
        allowed_cols_clinical = ['subject','Treatment_group','Baseline_HDRS21_totalscore',' Baseline_HARS_totalscore','6-weeks HARS_totalscore','6-weeks_HDRS21_totalscore']
    else:
        allowed_cols_clinical = ['Treatment_group','Baseline_HDRS21_totalscore',' Baseline_HARS_totalscore','6-weeks HARS_totalscore','6-weeks_HDRS21_totalscore']
    
    restricted_cols_clinical = get_set_without_items(clinical_col_set, items = allowed_cols_clinical)
    restricted_cols_bna = set(['site','subject.elm_id','ageV1','taskData.elm_id','visit','taskData.acqSysId','taskClass.elm_id','key']) #'remove added to visits
    cols_to_drop = restricted_cols_bna.union(restricted_cols_clinical) #all the columns we want to remove 
    subjects_baseline = drop_cols(subjects_baseline, cols_to_drop) #dropping 
    #bna.to_csv('second_research/subjects_baseline.csv',index = False) #26/11 lost gamma
    return subjects_baseline
    