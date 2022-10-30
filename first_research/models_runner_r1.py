import array
from heapq import merge
from itertools import groupby
from locale import format_string
from pathlib import Path
from matplotlib import pyplot as plt, test
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt
import pandas as pd
import numpy as np
import datetime as dt
import seaborn as sns
import warnings
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score
from sklearn.neighbors import kneighbors_graph
from X_initializer_r1 import get_c_to_group_by_c_dict, get_subject_change_rate_in_column_c_from_visit_i_to_visit_j
warnings.simplefilter("ignore", UserWarning)
#import the_module_that_warns

# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)

 

def print_conclusions(model_name,K_best_features,y_name, score, data_size,features_number):
    """Given a tested model, supplying conclusion of the train and prediction"""
    print("______________________________________________________________")
    print("*** CONCLUSIONS OF MODEL TRAINING AND TESTING: \n***")
    print("MODEL NAME: ", model_name,'\n')
    print("DATA SIZE (TOTAL OBSERVATIONS NUMBER: TRAIN + TEST): ", data_size,'\n')
    print("'X' BEST K FEATURES NUMBER: ", features_number,'\n') #size of X vector
    print("'X' BEST K FEATURES: ",'\n',K_best_features,'\n')
    print("PREDICTED 'y' VALUE: ", y_name,'\n') # y column name
     
    if model_name == "lr":
        # score func in lr is r2_score (r squared). more info in wikipedia
        score_func = "r2_score (r squared)"
        print("SCORE FUNCTION: ", score_func,'\n')
        if score < 0.5:
            print(f"SCORE: {score}, too low (the closer to 1, the better the model is),'\n'")
        else:
            print("SCORE = ", score,'\n')
    

    if model_name  == 'knn':
         # score func in knn is 'knn.score'- calculates the ratio of correct predictions of all predictions
        score_func = 'knn_score- ratio of ( # correct predictions (y_predict) / all right values (y_test)'
        print("SCORE FUNCTION: ", score_func,'\n')
        print("SCORE: ",str(score*100)+'%'+" ACCURACY IN PREDICTION",'\n')
    
    print("______________________________________________________________")    

def get_subject_end_of_treatment_state_in_column_c(subject_df,c):
    """a 'y' column creation function.
    Given a df of a particular subject(in sorted order of visits) and a column name c, returning the end-of-expirement value
    for this subject in column c """
    num_of_visits = len(subject_df)
    ans = subject_df.iloc[num_of_visits-1][c] # the value of the last visit in the column named c
    return ans

def convert_HDRS_17_score_to_class(HDRS_17_score,HDRS_17_change_rate):
    """Given an HDRS-17 score, returning a proper class to it. note that the classes computed 
    according to page 1 scoring part at this article https://dcf.psychiatry.ufl.edu/files/2011/05/HAMILTON-DEPRESSION.pdf
    """
    ans = None
    #print((HDRS_17_score,HDRS_17_change_rate))
    if 0 <= HDRS_17_score <= 7:
        ans = "remission" #also called normal state
    else:
        if HDRS_17_change_rate < -50:
            ans = "responsive"
        else:
            ans = "non_responsive"
    
    #print("ans", ans)
    return ans

 
def lr(subject_to_subject_group,subjects_X,k_select_k_best, y_name):
    """Training and reporting scoring results for the linear regression model.
    """
    #Step 1: adding the y column:
    if y_name == 'change_HDRS-17':
        subjects_X[y_name] = subjects_X['subject'].apply(lambda subject: get_subject_change_rate_in_column_c_from_visit_i_to_visit_j(subject_to_subject_group[subject],c='HDRS-17',i=0,j=len(subject_to_subject_group[subject])-1))
        subjects_X = subjects_X[subjects_X[y_name]!= 0] #drop out subjects with no change (meaning they have problems in data)
    #Step 2: remove y column and 'subject' from 'X'
    X = subjects_X.drop(columns= ['subject',y_name]) # at the moment we drop subject right before starting training the model. this is not the most beaufitul design and it caused beacuse we work with 'subejct_to_subject_group' dictionary. 
    # we can somehow prevent this ugly code by maybe by changing the df 'visits' (subjects_X in this case) to a new df with: row == subject (without cols like subject number or name which is redundant) and and columns will be best k features (relevant for any model) and the y column
    y = subjects_X[y_name]
    X.fillna(0, inplace = True) #design it with less code duplication
    y.fillna(0, inplace = True)
    #Step 3: select k best features from the 'X' vector, by the 'MRMR' (maximun relevancy, minimun redundancy) principal:
    #for the actual machine learning model:
    #save X_new, without names of columns:
    selector = SelectKBest(score_func=f_regression,k=k_select_k_best) #'best' = have the highest correllation with 'y' comparing to others.
    X_new = selector.fit_transform(X,y) # best k features from X, without names
    #save X_new, but with names of columns:
    cols = selector.get_support(indices=True)
    X_new = X.iloc[:,cols] #best k features of X, with their original names
    X_new_y = X_new.join(y) # best k features and y (in order to comput corellation between each one of them and the y)
    
    # not a step, but saving and showing plots and corellations:
    # abs_correlations = abs(X_new_y.corr()[y_name])
    # abs_correlations.rename('abs_corr_with_y', inplace=True)
    # print("top k highest features in corellation to y and their abs correllations:")
    # print(abs_correlations)
    # fig_dims = (10,5)
    # fig,ax = plt.subplots(figsize=fig_dims)
    # sns.heatmap(X_new_y.corr(),ax=ax)
    # plt.draw()

    # Step 4: WE need to add here some solution for the minimun redundadce part:
    #... COMPLETE HERE...
    # Step 5: split data for training and testing
    # create correlations hit map :
    X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size=0.2)
    # Step 6:: train model with fit
    regressor = LinearRegression()
    trained_model = regressor.fit(X_train, y_train) 
    # Step 7: predict X_test y scores (y_prediction) and compare them to real y scores (y_test)
    r2_score = regressor.score(X_test,y_test)  # doing both prediction and r2 calculation
    print_conclusions(model_name= "lr",K_best_features=X_new.columns.values, y_name = y_name,score=r2_score,data_size = X_new.shape[0],features_number = X_new.shape[1])
    print("finished running with no bugs")
    # plt.show() #enteryin an infinite while

  
def knn(subject_to_subject_group,subjects_X,k_select_k_best,k_knn, y_name):
    """Training and reporting scoring results for the knn model"""
    #Step 1: adding the y column:
    print("0")
    if y_name == 'end_of_treatment_class_HDRS-17':
        #Step 1: build y:
        subjects_X['change_HDRS-17'] = subjects_X['subject'].apply(lambda subject: get_subject_change_rate_in_column_c_from_visit_i_to_visit_j(subject_to_subject_group[subject],c='HDRS-17',i=0,j=len(subject_to_subject_group[subject])-1))
        d =  get_c_to_group_by_c_dict(subjects_X,'subject')
        #y:
        subjects_X[y_name] = subjects_X['subject'].apply(lambda subject: convert_HDRS_17_score_to_class(get_subject_end_of_treatment_state_in_column_c(subject_to_subject_group[subject],c='HDRS-17'),get_subject_end_of_treatment_state_in_column_c(d[subject],c='change_HDRS-17')))
        #print(subjects_X) 
    #Step 2: remove y, change in HDRS  column and 'subject' coulumn from 'X'
    print(subjects_X.head)
    X = subjects_X.drop(columns=['subject','change_HDRS-17',y_name]) # at the moment we drop subject right before starting training the model. this is not the most beaufitul design and it caused beacuse we work with 'subejct_to_subject_group' dictionary. 
    print(X.head)
    # we can somehow prevent this ugly code by maybe by changing the df 'visits' (subjects_X in this case) to a new df with: row == subject (without cols like subject number or name which is redundant) and and columns will be best k features (relevant for any model) and the y column
    y = subjects_X[y_name]
    X.fillna(0, inplace = True)
    y.fillna(0, inplace = True)
    #Step 3: select k best features from the 'X' vector, by the 'MRMR' (maximun relevancy, minimun redundancy) principal:
    #for the actual machine learning model:
    #save X_new, without names of columns:
    print(X.dtypes)
    print(X.head)
    print([col for col in X.columns if col.dtype == object])

    ### 29.10 a bug somewhere here. it seems like it doesent thik all df[i][j] are numbers so it cannot pass it to fit.
    
    selector = SelectKBest(score_func=f_classif,k=k_select_k_best) #'best' = have the highest correllation with 'y' comparing to others.
    X_new = selector.fit_transform(X,y) # best k features from X, without names
    #save X_new, but with names of columns:
    cols = selector.get_support(indices=True)
    X_new = X.iloc[:,cols] #best k features of X, with their original names
    X_new_y = X_new.join(y) # best k features and y (in order to comput corellation between each one of them and the y)
    print(X_new_y.head())
    # Step 4: WE need to add here some solution for the minimun redundadce part:
    #... COMPLETE HERE...
    # Step 5: split data for training and testing

    print("1")
    X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size=0.2)
    print("2")
    # Step 6:: train model with fit
    classifier = KNeighborsClassifier(n_neighbors=k_knn)
    trained_model= classifier.fit(X_train, y_train) 
    # Step 7: predict X_test y scores (y_prediction) and compare them to real y scores (y_test)
    knn_score = classifier.score(X_test,y_test) #Return the mean accuracy on the given test data and labels.
    y_prediction = trained_model.predict(X_test)
    print("prediction of model values: \n", y_prediction)
    print("real class values (y_test): \n ", y_test)
    print_conclusions(model_name= "knn", K_best_features= X_new.columns.values,y_name = y_name, score=knn_score,data_size = X_new.shape[0],features_number = X_new.shape[1])
    print("finished running with no bugs")
   

