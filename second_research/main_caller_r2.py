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


def select_best_model(X_train, y_train)->tuple:
    """Given X and y (dfs), returning  a 2-sized tuple (model,score) where model is the model name and score is an accuracy score,
    out of a dictionary models_to_scores """
    models_to_scores = get_model_names_to_model_accuracy_scores_dict(X_train,y_train)
    #selecting the model with the highest score:
    tmp_score = -1 #tmp_best_score
    for model in models_to_scores:
        configuration_list  = models_to_scores[model]
        for configuration in configuration_list:
            score = configuration[0]
            if score > tmp_score: # found a higher model score than we had
                tmp_score = score
                tmp_configuration = configuration
                tmp_model = model
    ans = (tmp_model,tmp_configuration)
    print(ans)
    return ans


def get_model_names_to_model_accuracy_scores_dict(X,y)->dict:
    """Given X and y (dfs), returing a dictionary of key = model name and val =average model accuracy score, after trying a few
    classification models to train X on y, with k-fold cross validation for each model."""

    model_name_to_configurations = {"dt":[], "knn":[]}
    k_fold_validation = KFold(10) # 10 fold validation
    
    for model_name in model_name_to_configurations: #iterate over keys  
        for k in range(2,20): # k  of select k best
            #Select k best features:
            selector = SelectKBest(score_func=f_classif,k=k) 
            selector.fit_transform(X,y) # best k features from X, without names
            #best k features of X, with their original names (technical part only):
            cols = selector.get_support(indices=True)
            X_copy = X.iloc[:,cols] #X after selelect k best with the original column names again
            #Train model on k selected features with cross validation and save the result and arguments(avg_score,k,n) in the dictionary:
            if model_name == 'knn': # k nearest neighbors classifier
                for neighbors_num in range(1,10):
                    model = KNeighborsClassifier(neighbors_num) 
                    results = cross_val_score(model,X_copy,y,cv=k_fold_validation) # list of k accuracy scores, each score of a trial on a different partition to train and test
                    average_model_score = np.mean(results)
                    configuration = (average_model_score, k,neighbors_num, list(X_copy.columns))
                    model_name_to_configurations[model_name].append(configuration) #acerage model scores in k fold cv
            
            if model_name == 'dt':
                model = DecisionTreeClassifier() # decision tree classifier
                results = cross_val_score(model,X_copy,y,cv=k_fold_validation) # list of k accuracy scores, each score of a trial on a different partition to train and test
                average_model_score = np.mean(results)
                configuration = (average_model_score,k, list(X_copy.columns))
                model_name_to_configurations[model_name].append(configuration) #acerage model scores in k fold cv

    return model_name_to_configurations
     


def get_y_column(X,y_name):
    """Given a df X and a string y_name, returning a column y of it, corresponding to X rows"""

    if y_name == '6-weeks_HDRS21_class': 
        base = 'Baseline_HDRS21_totalscore'
        end = '6-weeks_HDRS21_totalscore'     
        X = X[X[end].notna()] 

    if y_name == '6-weeks_HARS_class':
        base = ' Baseline_HARS_totalscore'
        end = '6-weeks HARS_totalscore'  
    
    change = '%_change_rate'
    X[change] = ((X[end] - X[base]) / X[base])* 100  # increase in % in  total score 
    #print("Avergage change rate = ", X[change].mean())
    #print("Num of subjects with target score = ",X.shape[0]) 
    y = pd.DataFrame()
    y[y_name] = X.apply(lambda k: convert_change_rate_to_class(k[change],k[end]),axis=1) #apply function using more than one column.https://stackoverflow.com/questions/13331698/how-to-apply-a-function-to-two-columns-of-pandas-dataframe
    #print("counts of y's y_name column" 
    #print(y[y_name].value_counts())
    X.drop(change,axis=1,inplace=True) # 1650 cols to 1649. remove col from X since it was just a helpfer col to get out y col (and can effect prediction negatively)
    return y
    

    
def convert_change_rate_to_class(change_rate,end):
    """Given change rate in % from baseline and week 6 score, return the class(state) due to that score and change rate """
    state = None
    if change_rate < -50: # drop of 50 percents of more from the baseline
        state = "responsive"
    else:
        state = "non_responsive"
    return state


 

def specify_X_to_y(X,y_name):
    """Based on the y, Removing redundant columns and subjects with missing values in this y"""

    if y_name == '6-weeks_HDRS21_class':
        end = '6-weeks_HDRS21_totalscore'  
    if y_name == '6-weeks_HARS_class':
        end = '6-weeks HARS_totalscore' 
    X= X[X[end].notna()] #removing missing values: keeping only subjects with not-None 'end' values (end = week 6 HDRS score or week 6 HARS score)  
    X.to_csv('second_research/X_specific_check_if_contains_missing_vals.csv',index = False)
    return X
    
#*Main*:
def main(y_name): #y_name = '6-weeks_HDRS21_class' or '6-weeks_HARS_class'
    bna_path = 'second_research\EYEC_Cordance_Gamma.csv'
    clinical_path = 'second_research\BW_clinical_data.csv'
    # Step 1: prepare X for model training
    X = numeric_df_initializer.generate_prediction_df(bna_path,clinical_path) #initial numeric-predicting data frame
    #print("y = ", y_name)
    X_specific = specify_X_to_y(X.copy(),y_name) # remove the second y  columns  
    y = get_y_column(X_specific,y_name) 
    X_specific.drop(['6-weeks_HDRS21_totalscore','6-weeks HARS_totalscore','Baseline_HDRS21_totalscore',' Baseline_HARS_totalscore'],axis=1,inplace=True) #we dont want week 6 features to effect the prdiction
    X_train, X_test,y_train, y_test = train_test_split(X_specific, y, test_size=0.2,random_state = 0) #rs is seed
    # split ceated successfuly 31.10 (82 train/21 test/103 total (I think))
    # Step 2: select classification model with k fold cross validation  
    selected_model = select_best_model(X_train,y_train) # selecting best classification model with cross validation. using 80% of data
    # one shot preditciton:
    selected_model.fit(X_train,y_train)
    y_pred = selected_model.predict(X_test) # real 1 shot prediction
    selected_model_accuracy_score = metrics.accuracy_score(y_test,y_pred)
#main('6-weeks_HARS_class')
#main('6-weeks_HDRS21_class')


 
def get_X_y (y_name): # for use in GSCVrunner
    bna_path = 'second_research\EYEC_Cordance_Gamma.csv'
    clinical_path = 'second_research\BW_clinical_data.csv'
    # Step 1: prepare X for model training
    X = numeric_df_initializer.generate_prediction_df(bna_path,clinical_path) #initial numeric-predicting data frame
    #print("y = ", y_name)
    X_specific = specify_X_to_y(X.copy(),y_name) # remove the second y  columns  
    y = get_y_column(X_specific,y_name) 
    X_specific.drop(['6-weeks_HDRS21_totalscore','6-weeks HARS_totalscore','Baseline_HDRS21_totalscore',' Baseline_HARS_totalscore'],axis=1,inplace=True) #we dont want week 6 features to effect the prdiction
    return X_specific,y