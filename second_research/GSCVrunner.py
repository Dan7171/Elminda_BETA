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
from catboost import CatBoostClassifier
import warnings
import main_caller_r2
from sklearn.exceptions import DataConversionWarning
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import confusion_matrix, f1_score, recall_score
from sklearn.metrics import make_scorer
from sklearn.datasets import load_iris
from sklearn.utils import shuffle
np.seterr(divide='ignore', invalid='ignore')
warnings.filterwarnings(action='ignore')
warnings.simplefilter(action='ignore')
from sklearn.metrics import precision_score
from sklearn.base import BaseEstimator, TransformerMixin

from runArguments import args
def toy_data_run():

    # *** TOY DATA PROCESSING ***:
    # used that part of code to verify the pipeplines can scre well. they do. scores of randsearch cv's were around 0.95

    iris = load_iris()
    X_toy=iris.data
    y_toy=iris.target
    print(X_toy)
    print(y_toy)
    X_train, X_test,y_train, y_test = train_test_split(X_toy, y_toy, test_size=0.15,random_state = args["rs"]) 
    rand_search_1a_iris = RandomizedSearchCV(pipe1a,param1a,n_iter=args["n_iter"],refit=True,cv =args["cv"],verbose=1,random_state=args["rs"]).fit(X_train,y_train.ravel())
    print_conclusions("1a",rand_search_1a_iris)
    rand_search_1b_iris = RandomizedSearchCV(pipe1b,param1b,n_iter=args["n_iter"],refit=True,cv =args["cv"],verbose=1,random_state=args["rs"]).fit(X_train,y_train.ravel())
    print_conclusions("1b", rand_search_1b_iris)
    rand_search_2_iris = RandomizedSearchCV(pipe2,param2,n_iter=args["n_iter"],refit=True,cv =args["cv"],verbose=1,random_state=args["rs"]).fit(X_train,y_train.ravel())
    print_conclusions("2",rand_search_2_iris)
    rand_search_3_iris = RandomizedSearchCV(pipe3,param3,n_iter=args["n_iter"],refit=True,cv =args["cv"],verbose=1,random_state=args["rs"]).fit(X_train,y_train.ravel())
    print_conclusions("3",rand_search_3_iris)
    rand_search_4_iris = RandomizedSearchCV(pipe4,param4,n_iter=args["n_iter"],refit=True,cv =args["cv"],verbose=1,random_state=args["rs"]).fit(X_train,y_train.ravel())
    print_conclusions("4", rand_search_4_iris)
    rand_search_5_iris = RandomizedSearchCV(pipe5,param5,n_iter=args["n_iter"],refit=True,cv =args["cv"],verbose=1,random_state=args["rs"]).fit(X_train,y_train.ravel())
    print_conclusions("5", rand_search_5_iris)
    rand_search_6_iris = RandomizedSearchCV(pipe6,param6,n_iter=args["n_iter"],refit=True,cv =args["cv"],verbose=1,random_state=args["rs"]).fit(X_train,y_train.ravel())
    print_conclusions("6", rand_search_6_iris)

# to add as a transformer to the pipelines (dropping corellated features after kbest)
class CorrelationDropper(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.9):
        self.threshold = threshold
        
    def fit(self, X, y=None):
        # Compute the correlation matrix
        corr_matrix = np.corrcoef(X, rowvar=False)
        
        # Select upper triangle of correlation matrix
        upper = np.triu(np.ones(corr_matrix.shape), k=1)
        correlated_features = np.where(upper * np.abs(corr_matrix) > self.threshold)
        
        # Drop all but one of the correlated features
        self.to_drop = []
        for i, j in zip(*correlated_features):
            if i in self.to_drop:
                continue
            self.to_drop.append(j)
        
        return self
    
    def transform(self, X, y=None):
        # Drop the correlated features
        return np.delete(X, self.to_drop, axis=1)

#toy_data_run()
def print_conclusions(df,pipe,search):
    print("=== user arguments === \n \n ",args,'\n')
    print("=== Pipe === \n \n", str(pipe),'\n')
    # print("=== Pipe's classifier === \n\n",pipe[-1])
    print("=== best hyperparametes picked in cv (cv's best score) === \n \n",search.best_params_,'\n')
    # Get the feature names kBest selected (before corellation drop)
    print("=== feature selection proccess (cv's best score) === ")
    
    # get the features after select ones with high corellation to y value
    
    # if 'pca' in pipe.named_steps:
    #     selector = search.best_estimator_.named_steps['pca'] # Get the  transformer from the pipeline
    #     selected_features = df.columns[selector.get_support()].tolist()
    #     print("Pca selected: \n ", selected_features)

    if 'kBest' in pipe.named_steps:
        print("first feature selection (high corellation to y)")
        selector = search.best_estimator_.named_steps['kBest'] 
        selected_features = df.columns[selector.get_support()].tolist()
        print("K best selected: \n", selected_features)
    print('\n')
   
    # Get the final feature names of the estimator (after dropping features with corellation to each other)
    # if 'corr_drop' in pipe.named_steps:
    #     selector = search.best_estimator_.named_steps['corr_drop']
    #     selected_features = df.columns[selector.get_support()].tolist()
    #     print("second feature selection (after dropping features with high corellation to each other, the final features classifier trained on) : \n",selected_features)
    # Get the mean and standard deviation of the F1 score
    score_mean = search.cv_results_['mean_test_score'][search.best_index_]
    score_std = search.cv_results_['std_test_score'][search.best_index_]
    print("=== score (cv's best score): %.3f +/- %.3f === " % (score_mean, score_std),"\n\n")

    # # Get the mean and standard deviation of the precision, accuracy, and recall scores
    # precision_mean = search.cv_results_['mean_test_precision'][search.best_index_]
    # precision_std = search.cv_results_['std_test_precision'][search.best_index_]
    # accuracy_mean = search.cv_results_['mean_test_accuracy'][search.best_index_]
    # accuracy_std = search.cv_results_['std_test_accuracy'][search.best_index_]
    # recall_mean = search.cv_results_['mean_test_recall'][search.best_index_]
    # recall_std = search.cv_results_['std_test_recall'][search.best_index_]

    # print("Precision: %.3f +/- %.3f" % (precision_mean, precision_std))
    # print("Accuracy: %.3f +/- %.3f" % (accuracy_mean, accuracy_std))
    # print("Recall: %.3f +/- %.3f" % (recall_mean, recall_std))
    # print('\n')


# ==================== Offir's code: ===============
def CV_Score(y_true,y_pred):
    global yt,yp
    yt.append(y_true)
    yp.append(y_pred)
    cvscore = f1_score(y_true, y_pred)
    return cvscore

def scorer():
    return make_scorer(CV_Score)
# ===================================================
def run_all_cvs(exhaustive_grid_search,X_train, y_train):
    
    # train and test in CV for finding best classifier and classifier parameters: 
    # option 1: for faster results, use RandimizedSearchCV:
    # random cv searches in the greed: (picking configuragtions from the greed randonly n_iter times)
    if args["scoring_method"] != "all_scores": 
        scoring_method = args["scoring_method"]
    if args["scoring_method"] == "all_scores":
        pass
        # somehow needs to be here: scoring_method = get_scoring_method_for_all_scores() #(precision, recall, accuracy...)?
    rs = args["rs"]
    print(X)
    if not exhaustive_grid_search: #randomized
        rand_search_1a = RandomizedSearchCV(pipe1a,param1a,n_iter=args["n_iter"],cv =args["cv"],verbose=1 ,random_state= rs,scoring=args['scoring_method'], refit=True).fit(X_train.astype(float),y_train.astype(str))
 
        print_conclusions(X_train,pipe1a,rand_search_1a)
        rand_search_1b = RandomizedSearchCV(pipe1b,param1b,n_iter=args["n_iter"],cv =args["cv"],verbose=1,random_state=rs,scoring=args['scoring_method'], refit=True).fit(X_train.astype(float),y_train.astype(str))
        print_conclusions(X_train,pipe1b, rand_search_1b)
        rand_search_2 = RandomizedSearchCV(pipe2,param2,n_iter=args["n_iter"],cv =args["cv"],verbose=2,random_state= rs,scoring=args['scoring_method'], refit=True).fit(X_train.astype(float),y_train.astype(str))
        print_conclusions(pipe2,rand_search_2)
        rand_search_3 = RandomizedSearchCV(pipe3,param3,n_iter=args["n_iter"],cv =args["cv"],verbose=2,random_state= rs,scoring=args['scoring_method'], refit=True).fit(X_train.astype(float),y_train.astype(str))
        print_conclusions(pipe3,rand_search_3)
        rand_search_4 = RandomizedSearchCV(pipe4,param4,n_iter=args["n_iter"],cv =args["cv"],verbose=2,random_state= rs,scoring=args['scoring_method'], refit=True).fit(X_train.astype(float),y_train.astype(str))
        print_conclusions(pipe4, rand_search_4)
        rand_search_5 = RandomizedSearchCV(pipe5,param5,n_iter=args["n_iter"],cv =args["cv"],verbose=2,random_state= rs,scoring=args['scoring_method'], refit=True).fit(X_train.astype(float),y_train.astype(str))
        print_conclusions(pipe5, rand_search_5)
        rand_search_6 = RandomizedSearchCV(pipe6,param6,n_iter=args["n_iter"],cv =args["cv"],verbose=2,random_state= rs,scoring=args['scoring_method'], refit=True).fit(X_train.astype(float),y_train.astype(str))
        print_conclusions(pipe6, rand_search_6)
        rand_search_7 = RandomizedSearchCV(pipe7,param7,n_iter=20,refit=True,cv =args["cv"],verbose=0,random_state=rs,scoring=scoring_method).fit(X_train.astype(float),y_train.astype(str))
        print_conclusions(pipe7, rand_search_7)

    # Option 2: for accurate but slow results- use GridSearchCV:
    # Exhaustive greed search: trying all the configurations in the greed
    else: 
        search1a= GridSearchCV(pipe1b, param1b,cv =args["cv"],n_jobs=4,refit=True,scoring=scoring_method).fit(X_train, y_train.values.ravel()) 
        print_conclusions(search1a) 
        search1b= GridSearchCV(pipe1b, param1b, n_jobs=4,refit=True,scoring=scoring_method).fit(X_train, y_train.values.ravel()) 
        print_conclusions(search1b)
        search2 = GridSearchCV(pipe2, param2, n_jobs=4,refit=True,scoring=scoring_method).fit(X_train, y_train.values.ravel())  
        print_conclusions(search2)
        search3 = GridSearchCV(pipe3, param3, n_jobs=4,refit=True,cv =args["cv"],verbose=3,scoring=scoring_method).fit(X_train, y_train.values.ravel())  
        print_conclusions(search3)
        search4 = GridSearchCV(pipe4, param4, n_jobs=4,refit=True,cv =args["cv"],verbose=3,scoring=scoring_method).fit(X_train, y_train.values.ravel())
        print_conclusions(search4)
        search5 = GridSearchCV(pipe5, param5, n_jobs=4,refit=True,cv =args["cv"],verbose=3,scoring=scoring_method).fit(X_train, y_train.values.ravel())
        print_conclusions(search5)
        search6 = GridSearchCV(pipe6, param6, n_jobs=4,refit=True,cv =args["cv"],verbose=3,scoring=scoring_method).fit(X_train, y_train.values.ravel())
        print_conclusions(search6)
        search7 = GridSearchCV(pipe7,param7,n_jobs=4,refit=True,cv =args["cv"],verbose=0,scoring=scoring_method).fit(X_train,y_train.values.ravel())
        print_conclusions(search7)


    

# def drop_out_highly_correlated_features(df):
#     """https://stackoverflow.com/questions/29294983/how-to-calculate-correlation-between-all-columns-and-remove-highly-correlated-on"""  
#     threshold = 0.7
#     if args["drop_out_correlated"]:
#         # Create correlation matrix
#         corr_matrix = df.corr(method='pearson').abs()
#         print(corr_matrix)
#         # Select upper triangle of correlation matrix
#         upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
#         to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
#         # Drop features 
#         df.drop(to_drop, axis=1, inplace=True)
#         if args["debug"]:
#             print("head after drop out")
#             print(df.head)
#     return df


# *****************************************************************************************************
# ******************************************* MAIN ****************************************************
# *****************************************************************************************************

# chose both reseach1 + rreseach12 as train dta or only reseach12
if not args["both"]: # use research 2 only
    X,y = main_caller_r2.get_X_y('6-weeks_HDRS21_class',args["X_version"]) #X and y's creationa and processing
if args["both"]: # use both research 1 and research 2
    all_data = pd.read_csv('all_data.csv')
    X = all_data.iloc[:,:-1]
    y = all_data.iloc[: , -1:]

# dropping collumns with correlation over some threshold
# if args["drop_out_correlated"]:
#     if args["debug"]:
#         print("before drop out correlated columns: X shape = ", X.shape)
#     X = drop_out_highly_correlated_features(X) 
#     if args["debug"]:    
#         print("after drop out correlated columns: X shape = ", X.shape)

#shuffle order of rows to reduce noise in cross validation
shuffled = shuffle(X.join(y) , random_state=args["rs"])
shuffled = shuffled.reset_index(drop=True)
X=shuffled.iloc[:,:-1]
y= shuffled.iloc[: , -1:]


if args["age_under_50"]: # using only candidated under age of 50 
    df = X.join(y)
    df = df[df['age'] < 50]
    X = df.iloc[:,:-1]
    y = df.iloc[: , -1:]

# create the piplelines and greeds:

#models (not classifiers) to use in pipelines
pca = PCA()
scaler = StandardScaler()
kBest_selector = SelectKBest()

#classifiers (estimators for the piplene-final step)
clf1 = LogisticRegression(max_iter=10000,random_state=args["rs"])
clf2 = KNeighborsClassifier()
clf3 = SVC(probability=True, random_state=args["rs"])
clf4 = DecisionTreeClassifier(random_state=args["rs"])
clf5 = RandomForestClassifier(random_state=args["rs"])
clf6 = GradientBoostingClassifier(random_state=args["rs"])
clf7 = CatBoostClassifier(random_state=args["rs"], logging_level = 'Silent')
#The 'greeds' (dictionaries)
# note: parameters of different models pipelines can be set using '__' separated parameter names. modelname__parameter name = options to try ing gscv:
param1a = { #LOGISTIC REGRESSION with pca, no selectkbest 
    "pca__n_components": range(2,10),
    "classifier__C": np.logspace(-4, 4,50),
    "classifier__penalty": ['l1','l2'],
    "classifier" : [clf1]
}
param1b = {#LOGISTIC REGRESSION with selectkbest, no pca
    "classifier__C": np.logspace(-4, 4, 4), #classifier (logistic regression) param 'C' for tuning
    "kBest__k": [12], #selctKbest param 'k'for tuning. must be  <= num of features
    "kBest__score_func" : [mutual_info_classif,f_classif], #selctKbest param 'score_func'for tuning
    "classifier" : [clf1] # the classifier clf1 (LogisticRegression) will use as the final step in pipleine- the 'estimator'
}
param2 = { #KNN 
    "classifier__n_neighbors" : [5,10,15],
    "kBest__k": [i for i in range(4)],
    "kBest__score_func" : [mutual_info_classif,f_classif],
    "classifier" : [clf2]    
}
param3 = { #SVC
    'classifier__gamma': [0.1,0.5,1,1.5,2],#old =  [10,6,3,1, 0.1, 0.01, 0.001, 0.0001]-> selcted = 1, score =0.6529411764705882 
    'classifier__kernel': ['rbf','sigmoid','linear','poly'],
    'classifier__C':[0.1,1.10,100,1000],
    "kBest__k": [2,5,8,12,16,20,30,45], #  k should be smaller than num of features always 
    "kBest__score_func" : [mutual_info_classif,f_classif],
    "classifier" : [clf3]    
}
param4 = { # DECISION TREE
        'classifier__max_leaf_nodes': list(range(2, 100)), 
        'classifier__min_samples_split': [2, 3, 4], #reason I tried this classifier params https://medium.com/analytics-vidhya/decisiontree-classifier-working-on-moons-dataset-using-gridsearchcv-to-find-best-hyperparameters-ede24a06b489
            "kBest__k": [2,5,8,12,16,20,30,45],
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
        "kBest__k": [2,5,8,12,16,20,30,45],
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
param7 = { #CATBOOST CLASSIFIER 
    'classifier__depth': [3,7,10],
    'classifier__learning_rate': [0.1],
    "kBest__k": [5,10,15],
    "classifier": [clf7]
}

# tune parameters with gscv and train model:
pipe1a = Pipeline(steps=[("scaler", scaler), ("pca", pca),('corr_drop', CorrelationDropper(threshold=0.9)),("classifier", param1a["classifier"][0])])
pipe1b = Pipeline(steps=[("scaler", scaler), ("kBest",kBest_selector),('corr_drop', CorrelationDropper(threshold=0.9)),("classifier", param1b["classifier"][0])])
pipe2 = Pipeline(steps=[("scaler", scaler), ("kBest",kBest_selector),('corr_drop', CorrelationDropper(threshold=0.9)),("classifier", param2["classifier"][0])])
pipe3 = Pipeline(steps=[("scaler", scaler), ("kBest",kBest_selector),('corr_drop', CorrelationDropper(threshold=0.9)),("classifier", param3["classifier"][0])])
pipe4 = Pipeline(steps=[("scaler", scaler), ("kBest",kBest_selector),('corr_drop', CorrelationDropper(threshold=0.9)),("classifier",param4["classifier"][0])])
pipe5 = Pipeline(steps=[("scaler", scaler), ("kBest",kBest_selector),('corr_drop', CorrelationDropper(threshold=0.9)),("classifier", param5["classifier"][0])])
pipe6 = Pipeline(steps=[("scaler", scaler), ("kBest",kBest_selector),('corr_drop', CorrelationDropper(threshold=0.9)),("classifier", param6["classifier"][0])])
pipe7 = Pipeline(steps=[("scaler", scaler), ("kBest",kBest_selector),('corr_drop', CorrelationDropper(threshold=0.9)),("classifier", param7["classifier"][0])])



# Split:
if(args["split_rows"] == 1): # regular split=  don't drop subjects:    
    X_train, X_test,y_train, y_test = train_test_split(X, y, test_size=0.15,random_state = args["rs"]) 
    run_all_cvs(False,X_train,y_train)


if(args["split_rows"] == 2): # split by 'Treatment_group' (device - h1/h7)
    print("splitting by h1 h7 : ")

    #run seperately for each treatment group
    
    # for treatment group 1
    df1 = X[X['Treatment_group'] == 0].join(y, how = "inner")
    print("new data- only the rows where column 'Treatment_group is 0:") 
    X = df1.iloc[:, :-1]
    y = df1.iloc[:, -1]
    # print("X")
    # print(X)
    # print("y")
    # print(y)
    X_train, X_test,y_train, y_test = train_test_split(X, y, test_size=0.15,random_state = args["rs"]) 
    run_all_cvs(False, X_train,y_train)
    
    # for treatment group 2
    df2 = X[X['Treatment_group'] == 1].join(y, how = "inner")
    print("new data- only the rows where column 'Treatment_group is 1:") 
    X = df2.iloc[:, :-1]
    y = df2.iloc[:, -1]
    
    run_all_cvs(False, X_train,y_train)



    

# 6. 
#after that:
#1. choose hights score for i in all 'serchi' values
#since used refit=true, after picking best hyperparametes, 'searchi' after picking them- trained the model with those hyperparams on data train(x_train,y_train) 
#2. predict searchi.(X,y)

 

