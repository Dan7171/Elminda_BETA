from matplotlib import pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.svm import SVC
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectKBest, f_classif, f_regression, mutual_info_classif
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
import warnings
import main_caller_r2
from sklearn.exceptions import DataConversionWarning
np.seterr(divide='ignore', invalid='ignore')
warnings.filterwarnings(action='ignore')
warnings.simplefilter(action='ignore')

def print_conclusions(search):
    print("GRID SEARCH FOR HYPER-PARAMETERS TUNING- THE CONCLUSIONS:")
    
    print(search)
   
    print("BEST PARAMS: ",search.best_params_)
    print("BEST SCORE (AVG ACCURACY SCORE OF K FOLD CV ONLY ON X_train data): ", search.best_score_)
    

X,y = main_caller_r2.get_X_y(y_name='6-weeks_HDRS21_class')
X_train, X_test,y_train, y_test = train_test_split(X, y, test_size=0.2,random_state = 7) 
# inspired by:
#https://scikit-learn.org/stable/tutorial/statistical_inference/putting_together.html#open-problem-stock-market-structure

#models (not classifiers) to use in pipelines:
pca = PCA()
scaler = StandardScaler()
kBest_selector = SelectKBest()

#classifiers (estimators for the piplene-final step)
clf1 = LogisticRegression(max_iter=10000,random_state=7)
clf2 = KNeighborsClassifier()
clf3 = SVC(probability=True, random_state=7)
clf4 = DecisionTreeClassifier(random_state=7)
clf5 = RandomForestClassifier(random_state=7)
clf6 = GradientBoostingClassifier(random_state=7)

# Parameters of pipelines can be set using '__' separated parameter names:

param1a = { #LOGISTIC REGRESSION with pca, no selectkbest 
    "pca__n_components": [5, 15, 30, 45, 60],
    "classifier__C": np.logspace(-4, 4, 4),
    "classifier" : [clf1]
}

param1b = {#LOGISTIC REGRESSION with selectkbest, no pca
    "classifier__C": np.logspace(-4, 4, 4), #classifier (logistic regression) param 'C' for tuning
    "kBest__k": [5,10,15], #selctKbest param 'k'for tuning
    "kBest__score_func" : [mutual_info_classif,f_classif], #selctKbest param 'score_func'for tuning
    "classifier" : [clf1] # the classigier clf1 (LogisticRegression) will use as the final step in pipleine- the 'estimator'
}
param2 = { #KNN 
    "classifier__n_neighbors" : [5,10,15],
    "kBest__k": [5,10,15],
    "kBest__score_func" : [mutual_info_classif,f_classif],
    "classifier" : [clf2]    
}
param3 = { #SVC
    'classifier__C': [0.1, 1, 10], #reason I tried this classifier params: https://www.geeksforgeeks.org/svm-hyperparameter-tuning-using-gridsearchcv-ml/ 
    'classifier__gamma': [1, 0.1, 0.01, 0.001, 0.0001],
    'classifier__kernel': ['rbf','sigmoid'],
    "kBest__k": [3,5,7], 
    "kBest__score_func" : [mutual_info_classif,f_classif],
    "classifier" : [clf3]    
}
param4 = { # DECISION TREE
           'classifier__max_leaf_nodes': list(range(2, 100)), 
           'classifier__min_samples_split': [2, 3, 4], #reason I tried this classifier params https://medium.com/analytics-vidhya/decisiontree-classifier-working-on-moons-dataset-using-gridsearchcv-to-find-best-hyperparameters-ede24a06b489
            "kBest__k": [5,10,15],
            "kBest__score_func" : [mutual_info_classif,f_classif],
            "classifier" : [clf4]   
}
param5 = { # RANDOM FOREST 
# reason I tried this classifier params:
# https://towardsdatascience.com/hyperparameter-tuning-the-random-forest-in-python-using-scikit-learn-28d2aa77dd74
        'bootstrap': [True],
        'max_depth': [10, 50, 100, 200],
        'max_features': [2,5,10,20,50],
        'min_samples_leaf': [2, 5,10],
        'min_samples_split': [2,3,4],
        'n_estimators': [10,50,80,120,300,1000], #reason I tried this classifier params https://medium.com/analytics-vidhya/decisiontree-classifier-working-on-moons-dataset-using-gridsearchcv-to-find-best-hyperparameters-ede24a06b489
            "kBest__k": [5,10,15],
            "kBest__score_func" : [mutual_info_classif,f_classif],
            "classifier" : [clf5]   
}
from sklearn.model_selection import RandomizedSearchCV

# tune parameters with gscv and train model:
pipe1a = Pipeline(steps=[("scaler", scaler), ("pca", pca),("classifier", param1b["classifier"][0])])
pipe1b = Pipeline(steps=[("scaler", scaler), ("kBest",kBest_selector),("classifier", param1b["classifier"][0])])
pipe2 = Pipeline(steps=[("scaler", scaler),("kBest",kBest_selector),("classifier", param2["classifier"][0])])
pipe3 = Pipeline(steps=[("scaler", scaler),("kBest",kBest_selector),("classifier", param3["classifier"][0])])
pipe4 = Pipeline(steps=[("scaler", scaler),("kBest",kBest_selector),("classifier",param4["classifier"][0])])
pipe5 = Pipeline(steps=[("scaler", scaler),("kBest",kBest_selector),("classifier", param5["classifier"][0])])
# search1a = GridSearchCV(pipe1a, param1a, n_jobs=2,refit=True).fit(X_train, y_train.values.ravel()) 
# search1b= GridSearchCV(pipe1b, param1b, n_jobs=2,refit=True).fit(X_train, y_train.values.ravel())  
# search2 = GridSearchCV(pipe2, param2, n_jobs=2,refit=True).fit(X_train, y_train.values.ravel())  
#search3 = GridSearchCV(pipe3, param3, n_jobs=-1,refit=True,cv=3,verbose=3).fit(X_train, y_train.values.ravel())  
#search4 = GridSearchCV(pipe4, param4, n_jobs=-1,refit=True,cv=3,verbose=3).fit(X_train, y_train.values.ravel())
rand_search_5 = RandomizedSearchCV(pipe5,param5,n_iter=30,n_jobes=-1,refit=True,cv=3,verbose=2,random_state=7).fit(X_train,y_train.values.ravel())
print(rand_search_5.best_params_)

# print_conclusions(search1a)
# print_conclusions(search1b)
# print_conclusions(search2)
#print_conclusions(search3)
#print_conclusions(search4)

#1. choose hights score for i in all 'serchi' values
#since used refit=true, after picking best hyperparametes, 'searchi' after picking them- trained the model with those hyperparams on data train(x_train,y_train) 
#2. predict searchi.(X,y)
