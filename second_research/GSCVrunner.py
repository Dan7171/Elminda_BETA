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
from sklearn.model_selection import RandomizedSearchCV
np.seterr(divide='ignore', invalid='ignore')
warnings.filterwarnings(action='ignore')
warnings.simplefilter(action='ignore')

def print_conclusions(search):
    print("HYPERPARAMETERS TUNING CONCLUSIONS:")
    print("BEST PARAMS: ",search.best_params_)
    print("BEST SCORE (MEAN ACCURACY SCORE OF K FOLD CV ONLY ON X_train data): ", search.best_score_)
    #print("ALL SCORES:", search.cv_results_)
    print('\n')

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

#The 'greeds' (dictionaries)
# note: parameters of different models pipelines can be set using '__' separated parameter names. modelname__parameter name = options to try ing gscv:
param1a = { #LOGISTIC REGRESSION with pca, no selectkbest 
    "pca__n_components": range(2,35),
    "classifier__C": np.logspace(-4, 4,50),
    "classifier__penalty": ['l1','l2'],
    "classifier" : [clf1]
}
param1b = {#LOGISTIC REGRESSION with selectkbest, no pca
    "classifier__C": np.logspace(-4, 4, 4), #classifier (logistic regression) param 'C' for tuning
    "kBest__k": [5,7,10,15,20,25,30,50,75,100], #selctKbest param 'k'for tuning
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
        'classifier__bootstrap': [True],
        'classifier__max_depth': [10, 50, 100, 200],
        'classifier__max_features': [2,5,10,20,50],
        'classifier__min_samples_leaf': [2, 5,10],
        'classifier__min_samples_split': [2,3,4],
        'classifier__n_estimators': [10,50,80,120,300,1000], #reason I tried this classifier params https://medium.com/analytics-vidhya/decisiontree-classifier-working-on-moons-dataset-using-gridsearchcv-to-find-best-hyperparameters-ede24a06b489
        "kBest__k": [5,10,15],
        "kBest__score_func" : [mutual_info_classif,f_classif],
        "classifier" : [clf5]   
}

param6 = { #GRADIENT BOOSTING 
# reason I tried this classifier params:
#https://www.analyticsvidhya.com/blog/2016/02/complete-guide-parameter-tuning-gradient-boosting-gbm-python/
    'classifier_max_depth':range(5,16,2),
    'classifier_min_samples_split':range(200,1001,200),
    "classifier": [clf6]
}

# tune parameters with gscv and train model:
pipe1a = Pipeline(steps=[("scaler", scaler), ("pca", pca),("classifier", param1a["classifier"][0])])
pipe1b = Pipeline(steps=[("scaler", scaler), ("kBest",kBest_selector),("classifier", param1b["classifier"][0])])
pipe2 = Pipeline(steps=[("scaler", scaler),("kBest",kBest_selector),("classifier", param2["classifier"][0])])
pipe3 = Pipeline(steps=[("scaler", scaler),("kBest",kBest_selector),("classifier", param3["classifier"][0])])
pipe4 = Pipeline(steps=[("scaler", scaler),("kBest",kBest_selector),("classifier",param4["classifier"][0])])
pipe5 = Pipeline(steps=[("scaler", scaler),("kBest",kBest_selector),("classifier", param5["classifier"][0])])
pipe6 = Pipeline(steps=[("scaler", scaler),("kBest",kBest_selector),("classifier", param6["classifier"][0])])


#random cv searches in the greed: (picking configuragtions from the greed randonly n_iter times)

#rand_search_1a = RandomizedSearchCV(pipe1a,param1a,n_iter=10,refit=True,cv=5,verbose=1,random_state=7).fit(X_train,y_train.values.ravel())
#print_conclusions(rand_search_1a)
# rand_search_1b = RandomizedSearchCV(pipe1b,param1b,n_iter=10,refit=True,cv=10,verbose=1,random_state=7).fit(X_train,y_train.values.ravel())
# print_conclusions(rand_search_1b)
# rand_search_2 = RandomizedSearchCV(pipe2,param2,n_iter=10,refit=True,cv=3,verbose=1,random_state=7).fit(X_train,y_train.values.ravel())
# print_conclusions(rand_search_2)
# rand_search_3 = RandomizedSearchCV(pipe3,param3,n_iter=10,refit=True,cv=3,verbose=1,random_state=7).fit(X_train,y_train.values.ravel())
# print_conclusions(rand_search_3)
rand_search_4 = RandomizedSearchCV(pipe4,param4,n_iter=10,refit=True,cv=3,verbose=1,random_state=7).fit(X_train,y_train.values.ravel())
print_conclusions(rand_search_4)
# rand_search_5 = RandomizedSearchCV(pipe5,param5,n_iter=10,refit=True,cv=3,verbose=1,random_state=7).fit(X_train,y_train.values.ravel())
# print_conclusions(rand_search_5)
# rand_search_6 = RandomizedSearchCV(pipe6,param6,n_iter=10,refit=True,cv=3,verbose=1,random_state=7).fit(X_train,y_train.values.ravel())
# print_conclusions(rand_search_6)


# Exhaustive greed search: trying all the configurations in the greed
#search1a= GridSearchCV(pipe1b, param1b,cv=5,n_jobs=4,refit=True).fit(X_train, y_train.values.ravel()) 
#print_conclusions(search1a) 
#search1b= GridSearchCV(pipe1b, param1b, n_jobs=4,refit=True).fit(X_train, y_train.values.ravel()) 
#print_conclusions(search1b)
#search2 = GridSearchCV(pipe2, param2, n_jobs=4,refit=True).fit(X_train, y_train.values.ravel())  
#print_conclusions(search2)
#search3 = GridSearchCV(pipe3, param3, n_jobs=4,refit=True,cv=3,verbose=3).fit(X_train, y_train.values.ravel())  
#print_conclusions(search3)
search4 = GridSearchCV(pipe4, param4, n_jobs=4,refit=True,cv=3,verbose=3).fit(X_train, y_train.values.ravel())
print_conclusions(search4)
#search5 = GridSearchCV(pipe5, param5, n_jobs=4,refit=True,cv=3,verbose=3).fit(X_train, y_train.values.ravel())
#print_conclusions(search5)
#search6 = GridSearchCV(pipe6, param6, n_jobs=4,refit=True,cv=3,verbose=3).fit(X_train, y_train.values.ravel())
#print_conclusions(search6)

#after that:
#1. choose hights score for i in all 'serchi' values
#since used refit=true, after picking best hyperparametes, 'searchi' after picking them- trained the model with those hyperparams on data train(x_train,y_train) 
#2. predict searchi.(X,y)

