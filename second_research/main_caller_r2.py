


import numeric_df_initializer
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold 
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
import warnings
warnings.simplefilter(action='ignore')
warnings.filterwarnings(action='ignore')

from runArguments import args


def get_y_column(X,y_name):
    """Given a df X and a string y_name, returning a column y of it, corresponding to X rows"""

    base = 'Baseline_HDRS21_totalscore'
    end = '6-weeks_HDRS21_totalscore'     
    X = X[X[end].notna()] 
    
    change = '%_change_rate'
    X[change] = ((X[end] - X[base]) / X[base])* 100  # increase in % in  total score 
    y = pd.DataFrame()

    if args["classification"]: #classification
        y[y_name] = X.apply(lambda subject: convert_change_rate_to_class(subject[change]),axis=1) #apply function using more than one column.https://stackoverflow.com/questions/13331698/how-to-apply-a-function-to-two-columns-of-pandas-dataframe
    else: # regression
        y[y_name] = X[change].copy()
        
    X.drop(change,axis=1,inplace=True) # 1650 cols to 1649. remove col from X since it was just a helpfer col to get out y col (and can effect prediction negatively)
    return y
    

    
def convert_change_rate_to_class(change_rate):
    """Given change rate in % from baseline and week 6 score, return the class(state) due to that score and change rate """
    state = None
    if args["classification_type"] == "normal": # normal calssification, as was in the code all the time 
        if change_rate < -50: # drop of 50 percents of more from the baseline
            state = "responsive"
        else:
            state = "non_responsive"

    elif args["classification_type"] == "extreme": # super responders vs super non responders
        if change_rate < -70: # drop of 50 percents of more from the
            state = "extremely_responsive"
        elif -70 <= change_rate <= -30:
            state = "extremely_not_responsive"
        else: # > -30
            state = "mid_range_responsive"
    return state


def specify_X_to_y(X,y_name):
    """Based on the y, Removing redundant columns and subjects with missing values in this y"""
    end = '6-weeks_HDRS21_totalscore'  
    X= X[X[end].notna()] #removing missing values: keeping only subjects with not-None 'end' values (end = week 6 HDRS score or week 6 HARS score)  
    return X
    
 
def get_X_y (y_name,X_version): # for use in GSCVrunner

    # Part a: generate X without filtering and y (y is same for X with or without filtering)
    if args['use_gamma_columns']:
        bna_path = 'EYEC_Cordance_Gamma_no_missing_vals.csv'
    else:
        bna_path = 'EYEC_Cordance_Gamma.csv'
    clinical_path = 'BW_clinical_data.csv'
    # prepare X for model training
    X = numeric_df_initializer.generate_prediction_df(bna_path,clinical_path) #initial numeric-predicting data frame
    X_specific = specify_X_to_y(X.copy(),y_name) # remove the second y  columns
    y = get_y_column(X_specific, y_name).reset_index(drop = True) # for technical reasons - y is always produced based on the original no-filte X (as seeing here), even if we chose to use filtered X (it's ok, no changing resuluts)
    X_specific.drop(['6-weeks_HDRS21_totalscore','6-weeks HARS_totalscore','Baseline_HDRS21_totalscore',' Baseline_HARS_totalscore'],axis=1,inplace=True) #we dont want week 6 features to effect the prdiction
    
    # Part b: replace X to be a filtered version if the argument X_version is not set to 1
    
    if not args['both']:
        # if X_version == 1: # no filtering, remain with the basic version
        #    do nothing
        if(X_version == 2): # basic filtering
            X_specific = pd.read_csv("X_(basic_filter).csv")
        if(X_version == 3): # hard filtering- predictors only
            X_specific = pd.read_csv("X_(predictors_only).csv")
    else: # use both research 1 and research 2,for now works in classification only
        all_data = pd.read_csv('all_data.csv')
        X_specific = all_data.iloc[:,:-1]
        y = all_data.iloc[: , -1:]
    return X_specific,y
    
