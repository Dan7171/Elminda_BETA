import numeric_df_initializer
import models_runner_r2
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold 
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.model_selection import cross_val_score
 
def select_best_model(X_train, y_train)->tuple:
    """Given X and y (dfs), returning  a 2-sized tuple (model,score) where model is the model name and score is an accuracy score,
    out of a dictionary models_to_scores """
    models_to_scores = get_model_names_to_model_accuracy_scores_dict(X_train,y_train)
    #selecting the model with the highest score:
    tmp = None
    for model in models_to_scores:
        score = models_to_scores[model]
        if (tmp == None) or (score > tmp): # found a higher model score than we had
            tmp = (model,score)
    return tmp 


def get_model_names_to_model_accuracy_scores_dict(X,y)->dict:
    """Given X and y (dfs), returing a dictionary of key = model name and val =average model accuracy score, after trying a few
    classification models to train X on y, with k-fold cross validation for each model."""
    model_to_accuracy = {"dt":-1, "knn":-1}
    k_fold_validation = KFold(10) # 10 fold validation
    for k in range(2,5): # k of select best k
        for model in model_to_accuracy: 
            if model == 'knn': # k nearest neighbors classifier
                for neighbors_num in range(1,5):
                    model = KNeighborsClassifier(X,y,n_neighbors=neighbors_num) #
            if model == 'dt':
                model = DecisionTreeClassifier() # decision tree classifier
            results = cross_val_score(model,X,y,cv=k_fold_validation) # list of k accuracy scores, each score of a trial on a different partition to train and test
            average_model_score = np.mean(results)
            model_to_accuracy[model] = np.mean(average_model_score) #acerage model scores in k fold cv
    
    return model_to_accuracy



def get_y_column(X,y_name):
    """Given a df X and a string y_name, returning a column y of it, corresponding to X rows"""

    if y_name == '6-weeks_HDRS21_class': 
        base = 'Baseline_HDRS21_totalscore'
        end = '6-weeks_HDRS21_totalscore'     
        X = X[X[end].notna()] 

    if y_name == '6-weeks_HARS_class':
        base = 'Baseline_HARS_totalscore'
        end = '6-weeks HARS_totalscore'  
    
    change = '%_change_rate'
    X[change] = ((X[end] - X[base]) / X[base])* 100  # increase in % in HDRS 21 total score 
    #print("Avergage change rate = ", X[change].mean())
    #print("Num of subjects with target score = ",X.shape[0]) 
    y = pd.DataFrame()
    y[y_name] = X.apply(lambda k: convert_change_rate_to_class(k[change],k[end]),axis=1) #apply function using more than one column.https://stackoverflow.com/questions/13331698/how-to-apply-a-function-to-two-columns-of-pandas-dataframe
    #print("counts of y's y_name column" 
    #print(y[y_name].value_counts())
    X.drop(change,axis=1,inplace=True) # 1650 cols to 1649. remove col from X since it was just a helpfer col to get out y col (and can effect prediction negatively)
    return y
    # finished

    
def convert_change_rate_to_class(change_rate,end):
    """Given change rate in % from baseline and week 6 score, return the class(state) due to that score and change rate """
    
    state = None
    if 0 <= end <= 7:
        state = "remission" #also called normal state
    else:
        if change_rate < -50: # drop of 50 percents of more from the baseline
            state = "responsive"
        else:
            state = "non_responsive"
    return state


 

def specify_X_to_y(X,y_name):
    """Based on the y, Removing redundant columns and subjects with missing values in this y"""

    if y_name == '6-weeks_HDRS21_class':
        end = '6-weeks_HDRS21_totalscore'
        cols_to_drop = ['6-weeks HARS_totalscore']

    if y_name == '6-weeks_HARS_class':
        end = '6-weeks HARS_totalscore' 
        cols_to_drop = ['6-weeks_HDRS21_totalscore']

    X= X[X[end].notna()] #removing missing values: keeping only subjects with not-None week 'end' values (end = week 6 HDRS score or week 6 HARS score)
    for col in cols_to_drop:
        X.drop(col,axis=1,inplace=True)
    print(X)
    print("just built this function")
    return X
    
#*Main*:
bna_path = 'second_research\EYEC_Cordance_Gamma.csv'
clinical_path = 'second_research\BW_clinical_data.csv'
# Step 1: prepare X for model training
X = numeric_df_initializer.generate_prediction_df(bna_path,clinical_path) #initial numeric-predicting data frame
y_names = ['6-weeks_HDRS21_class', '6-weeks_HARS_class']
for y_name in y_names:
    # using different X for each different y, because we drop from each X the subjects does not have this 
    X_specific = specify_X_to_y(X.copy(),y_name)  
    y = get_y_column(X_specific,y_name) 
    X_train, X_test,y_train, y_test = train_test_split(X_specific, y, test_size=0.2,random_state = 0) #rs is seed
    # split ceated successfuly 31.10 (82 train/21 test/103 total (I think))
    # Step 2: select classification model with k fold cross validation  
    
    print("continue to debug from here")
    selected_model = select_best_model(X_train,y_train ) # selecting best classification model with cross validation. using 80% of data
    # one shot preditciton:
    selected_model.fit(X_train,y_train)
    y_pred = selected_model.predict(X_test) # real 1 shot prediction
    selected_model_accuracy_score = metrics.accuracy_score(y_test,y_pred)







#X.to_csv('second_research/X_before prediction.csv',index = False)
#X_1 = X.copy()
# X_1['y'] = X_1['age']  - X_1['age']
# X_1['Z'] = X_1['age']  + X_1['y']
# print(X_1['y'])
# print(X_1['Z'])