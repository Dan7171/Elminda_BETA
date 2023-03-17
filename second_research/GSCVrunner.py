from matplotlib import pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.svm import SVC
import numpy as np
import pandas as pd
import sklearn
import datetime
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
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, roc_auc_score
from sklearn.metrics import make_scorer
from sklearn.datasets import load_iris
from sklearn.utils import shuffle
np.seterr(divide='ignore', invalid='ignore')
warnings.filterwarnings(action='ignore')
warnings.simplefilter(action='ignore')
from sklearn.base import BaseEstimator, TransformerMixin
import os
from scipy.stats import randint
from runArguments import args
warnings.filterwarnings("ignore", category=DeprecationWarning) 

def fxn():
    warnings.warn("deprecated", DeprecationWarning)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fxn()

yt,yp = [],[]
my_scorer = args['scoring_method']

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
class CorrelationDropper(BaseEstimator, TransformerMixin): # this one works, dont delete it
    def __init__(self, threshold=0.9):
        self.threshold = threshold
        
    def fit(self, X, y=None):
        # Compute the correlation matrix
        if args["drop_out_correlated"]:
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
        if args["drop_out_correlated"]:
            return np.delete(X, self.to_drop, axis=1)
        return X

class CorrelationDropper2(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.9):
        self.threshold = threshold
        self.indices_to_keep = []
    
    def _find_correlated_groups(self, correlations):
        correlated_groups = [None]*correlations.shape[1]
        visited_features = set()
        
        for i in range(correlations.shape[0]):
            marked = False
            for j in range(i+1, correlations.shape[1]):
                if abs(correlations[i, j]) > self.threshold:
                    marked = True
                    # Add correlated features to the same group
                    group = correlated_groups[i] if i in visited_features else correlated_groups[j] if j in visited_features else set()
                    group.add(i)
                    group.add(j)
                    visited_features.add(i)
                    visited_features.add(j)
                    correlated_groups[i] = group
                    correlated_groups[j] = group
            if not marked:
                group = set()
                group.add(i)
                visited_features.add(i)
                correlated_groups[i] = group
        
        print(correlated_groups)
        return correlated_groups
    
    def fit(self, X, y=None):
        if args["drop_out_correlated"]:
            # Calculate pairwise correlations between features
            correlations = np.corrcoef(X, rowvar=False)
            print(correlations)
            # Identify correlated groups
            correlated_groups = self._find_correlated_groups(correlations)
            
            # Drop all correlated features but one from each group
            self.indices_to_keep = []
            for group in correlated_groups:
                if len(group) >= 1:
                    self.indices_to_keep.append(list(group)[0])
            print(self.indices_to_keep)
        return self
    
    def transform(self, X, y=None):
        if args["drop_out_correlated"]:
            # Select desired features from array
            X_transformed = np.take(X, self.indices_to_keep, axis=1)
            return X_transformed
        return X
 
def print_conclusions(df,pipe,search,yt_cv=None,yp_cv=None):
    print("-----------------------\n New CV report \n-----------------------")
    print("* Classifier: \n", pipe.named_steps['classifier'])
    print("* User arguments: \n",args)
    print("* Pipeline details: \n" , str(pipe))
    print("* Best Hyperparametes picked in cross validation: (cv's best score): \n",search.best_params_)    
    
    # features Kbest selected
    selected_features = ""
    if 'kBest' in pipe.named_steps:
        selector = search.best_estimator_.named_steps['kBest'] 
        selected_features = df.columns[selector.get_support()].tolist()
    print("* Best features by (selectKbest): \n", selected_features)

    # score 
    print("* Scorer_used:", args['scoring_method']) # scoring method used for cv as scorer param
    score_mean = search.cv_results_['mean_test_score'][search.best_index_]
    score_std = search.cv_results_['std_test_score'][search.best_index_]
    print("* CV Score (cv's best score for best hyperparametes): %.3f +/- %.3f (see score func in hyperparams) " % (score_mean, score_std),"\n")


    cm = confusion_matrix(yt_cv, yp_cv)
    print("* Confusion matrix: \n",cm) 
    # confusion matrix plot making: 
    fig = metrics.ConfusionMatrixDisplay.from_predictions(yt_cv, yp_cv)
    fig.ax_.set_title(pipe.named_steps['classifier'])

    # calculate Response rate: 
    y_train_responders_cnt = 0
    positive_label = 1 # can vary in different expirements
    for y_val in y_train.values:
        if y_val == positive_label:
            y_train_responders_cnt += 1
    
    print("* Response rate: ",y_train_responders_cnt / len(y_train))
    true_count = cm[0][0] + cm[1][1]
    false_count = cm[0][1] + cm[1][0]
    total = true_count + false_count
    precision = cm[1][1] / (cm[1][1] + cm[0][1])
    recall = cm[1][1] / (cm[1][1] + cm[1][0])
    accuracy =  true_count /total
    f1 = 2 * (precision * recall) / (precision + recall)
    print("* CV Precision: ", precision)
    print("* CV Recall: ", recall)
    print("* CV Accuracy: ", accuracy)
    print("* CV F1: ",f1) 
   
    

    # save cross val scores and params in tuning.csv for tracking
    # csv column names and values in a row:
    d = {
    "date":str(datetime.date.today()), # date
    "classifier":str(pipe.named_steps['classifier']), 
    "pipe_named_steps": str(pipe.named_steps), # all pipe steps, including the classifier as estimator
    "best_params":str(search.best_params_), 
    "selected_features":str(selected_features),
    "user_inputs":str(args), # runArguments.args
    "responders_rate": str(((cm[1][0] + cm[1][1]) / total )), # responders / total
    "X_train_size(num of rows in cv input df)": str(total),
    "scorer_used": args['scoring_method'], # scoring method used for cv as scorer param
    "scorer_score_mean":str(score_mean),  # should be one of (accuracy,precission,recall,f1 (if scorer_used = f1))
    "scorer_score_std":str(score_std),
    "accuracy": accuracy,
    "precision": precision,
    "recall":recall,
    "f1":f1,
    "confusion_matrix":str(cm)
    }
    # save to cv:
    tmp = pd.DataFrame(d,index=[d.keys()])
    file_path = 'tuning.csv'
    if not os.path.exists(file_path):
    # If the file does not exist, create a new empty CSV file
        open(file_path, 'a').close()
        tmp.to_csv(file_path,index=False)     
        return
    df = pd.read_csv(file_path)
    df = pd.concat([df,tmp])
    df.to_csv(file_path,index=False)

    print("CV report saved to  ",file_path)
    print("-----------------------\n End of CV report \n-----------------------",'\n'*3)

 

def reset_column_to_original_labels(df,col,original_lables):
    df[col] = original_lables[df[col]]
    return df

 
# ==================== Offir's code: ===============
def CV_Score(y_true,y_pred):
    global my_scorer
    global yt,yp 
    yt.append(y_true) # y true
    yp.append(y_pred) # y predicted
    if my_scorer == 'roc_auc':
        cvscore = roc_auc_score(y_true, y_pred)
    if my_scorer == 'accuracy':
        cvscore = accuracy_score(y_true, y_pred)
        #cvscore = accuracy_score(y_true, y_pred,pos_label=1)
    if my_scorer == 'f1':
        cvscore = f1_score(y_true, y_pred)
    return cvscore

def scorer():
    return make_scorer(CV_Score)
# ===================================================
 


# *****************************************************************************************************
# ******************************************* MAIN ****************************************************
# *****************************************************************************************************


y_name = '6-weeks_HDRS21_class'

# chose both reseach1 + rreseach12 as train dta or only reseach12
if not args["both"]: # use research 2 only
    X,y = main_caller_r2.get_X_y(y_name,args["X_version"]) # X and y's creationa and processing

if args["both"]: # use both research 1 and research 2
    all_data = pd.read_csv('all_data.csv')
    X = all_data.iloc[:,:-1]
    y = all_data.iloc[: , -1:]

X.reset_index(inplace=True,drop=True)
y['6-weeks_HDRS21_class'] = y['6-weeks_HDRS21_class'].replace({"responsive": 1, "non_responsive": 0})

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

clf1 = LogisticRegression(random_state=args["rs"])
clf2 = KNeighborsClassifier()
clf3 = SVC(probability=True, random_state=args["rs"])
clf4 = DecisionTreeClassifier(random_state=args["rs"])
clf5 = RandomForestClassifier(random_state=args["rs"])
clf6 = GradientBoostingClassifier(random_state=args["rs"])
clf7 = CatBoostClassifier(random_state=args["rs"], logging_level = 'Silent')

#The param 'grids' 
# note: parameters of different models pipelines can be set using '__' separated parameter names. modelname__parameter name = options to try ing gscv:

param1a = { #LOGISTIC REGRESSION with pca, no selectkbest 
    "pca__n_components": range(2,50,3),
    "classifier__C": [0.001,0.01,0.1,1,10,100],
    "classifier__penalty": ['l2'],
    "classifier" : [clf1]
}
# tune 3. c=0.5, k = 16, penalty= l2,mutual_info
param1b = { #LOGISTIC REGRESSION with selectkbest, no pca
     
    "classifier__C":[0.45,0.47,0.5,0.52,0.55,0.6], #classifier (logistic regression) param 'C' for tuning
    "kBest__k": range(15,17), #selctKbest param 'k'for tuning. must be  <= num of features
    'classifier__penalty' : ['l2'],
    "kBest__score_func" : [mutual_info_classif], #selctKbest param 'score_func'for tuning
    "classifier" : [clf1]

}

param1b_offir_scoring_debug = { #LOGISTIC REGRESSION with selectkbest, no pca
    "classifier__C":[30], #classifier (logistic regression) param 'C' for tuning
    "kBest__k":[5], #selctKbest param 'k'for tuning. must be  <= num of features
    "kBest__score_func" : [f_classif], #selctKbest param 'score_func'for tuning
    "classifier" : [clf1] # the classifier clf1 (LogisticRegression) will use as the final step in pipleine- the 'estimator'
}
param2 = { #KNN 
    "classifier__n_neighbors" :range(1,90,3),
    "classifier__weights":['uniform','distance'],
    "classifier__algorithm": ['auto', 'ball_tree', 'kd_tree', 'brute'],
    "classifier__leaf_size": range(3,80,3),
    "classifier__p":[2],
    "kBest__k": range(4,100,3),
    "kBest__score_func" : [f_classif,mutual_info_classif], #selctKbest param 'score_func'for tuning
    "classifier" : [clf2]    
}
param3 = { #SVC
    'classifier__gamma':   [60,65,70,75,80], 
    'classifier__kernel': ['linear', 'rbf', 'sigmoid'],
    'classifier__C' : [900,950,1000,1100,1200,1400],
    "kBest__k": range(40,70,5), #  k should be smaller than num of features in X 
    "classifier" : [clf3]    
}
param4 = { # DECISION TREE
        'classifier__max_leaf_nodes': range(1,25,3), 
        'classifier__max_depth':[2,4,6,8,10,12],
        'classifier__criterion':['gini', 'entropy'],
        'classifier__min_samples_split': range(2,40,5), #reason I tried this classifier params https://medium.com/analytics-vidhya/decisiontree-classifier-working-on-moons-dataset-using-gridsearchcv-to-find-best-hyperparameters-ede24a06b489
        "kBest__k": range(4,40,8),
        "classifier" : [clf4]   
}
param5 = { # RANDOM FOREST 
# reason I tried this classifier params:
# https://towardsdatascience.com/hyperparameter-tuning-the-random-forest-in-python-using-scikit-learn-28d2aa77dd74
        'classifier__bootstrap': [True,False],
        "classifier__max_depth": range(2,50,2),
        "classifier__min_samples_split": range(2,50,3),
        "classifier__min_samples_leaf":  range(2,50,3),
        "classifier__max_features":  range(2,30,3),
        "kBest__k": range(4,80,3),
        "classifier" : [clf5]   
}

param6 = { #GRADIENT BOOSTING 
# reason I tried this classifier params:
#https://www.analyticsvidhya.com/blog/2016/02/complete-guide-parameter-tuning-gradient-boosting-gbm-python/
    "classifier__n_estimators":[5,10,30,50,100,150,250,400,500],
    'classifier__max_depth':range(4,80,3),
    "classifier__learning_rate":[0.01,0.1,1,10,100],
    "kBest__k":range(4,80,3),
    "classifier": [clf6]
}
param7 = { #CATBOOST CLASSIFIER 
    'classifier__n_estimators' : [3,10,30, 50, 100, 500],
    "classifier__learning_rate":[0.01,0.1,1,10,100],
    'classifier__subsample':[0.1,0.3,0.5, 0.7, 1.0],
    'classifier__max_depth': range(3,10,3),
    'classifier__learning_rate': [0.05,0.1,0.5],
    "kBest__k": range(4,40,8),
    "classifier": [clf7]
}

# define the pipelines
pipe1a = Pipeline(steps=[("scaler", scaler), ("pca", pca),("classifier", param1a["classifier"][0])])
pipe1b = Pipeline(steps=[("scaler", scaler), ("kBest",kBest_selector),("classifier", param1b["classifier"][0])])
pipe2 = Pipeline(steps=[("scaler", scaler), ("kBest",kBest_selector),("classifier", param2["classifier"][0])])
pipe3 = Pipeline(steps=[("scaler", scaler), ("kBest",kBest_selector),("classifier", param3["classifier"][0])])
pipe4 = Pipeline(steps=[("scaler", scaler), ("kBest",kBest_selector),("classifier",param4["classifier"][0])])
pipe5 = Pipeline(steps=[("scaler", scaler), ("kBest",kBest_selector),("classifier", param5["classifier"][0])])
pipe6 = Pipeline(steps=[("scaler", scaler), ("kBest",kBest_selector),("classifier", param6["classifier"][0])])
pipe7 = Pipeline(steps=[("scaler", scaler), ("kBest",kBest_selector),("classifier", param7["classifier"][0])])


#############
# 9.1 - try with less responsive subjets:
if args['balance_y_values']:
    data = X.join(y)
    data = data.drop(data[data['6-weeks_HDRS21_class'] == 1].sample(frac=.3).index)
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

############
splitted_congifs = [] # each list is a list of X_train, X_test,y_train, y_test to run cv and fit on

# Split data by rows into categories (or not):
if(args["split_rows"] == 'normal'): # regular test train split  =  don't drop subjects: 
    X_train, X_test,y_train, y_test = train_test_split(X, y, test_size=0.2,random_state = args["rs"],shuffle=True,stratify=y) 
    splitted_congifs.append([X_train, X_test,y_train, y_test])

if(args["split_rows"] in ['h1','h7','h1h7']): # split by 'Treatment_group' (device - h1/h7)
    print("splitting by h1 h7 : ")
    
    if args['split_rows'] in ['h1','h1h7']: #use h1
        # for treatment group 1 - h1 ('Treatment_group' is 0)
        # h1 subjects only
        df1= X.join(y)
        df1 = df1[df1['Treatment_group'] == 0]
        print("new data- only the rows where column 'Treatment_group is 0:") 
        X_tmp = df1.iloc[:, :-1]
        y_tmp = df1.iloc[:, -1]
        X_train, X_test,y_train, y_test = train_test_split(X_tmp, y_tmp, test_size=0.2,random_state = args["rs"],shuffle=True,stratify=y_tmp) 
        splitted_congifs.append([X_train, X_test,y_train, y_test])
    
    if args['split_rows'] in ['h7', 'h1h7']:   #use h7
        # for treatment group 2 - h7 ('Treatment_group' is 1) 
        # h7 subjects only
        # df2 = X[X['Treatment_group'] == 1].join(y, how = "inner") #H7
        df2 = X.join(y)
        df2 = df2[df2['Treatment_group'] == 1]
        print("new data- only the rows where column 'Treatment_group is 1:") 
        X_tmp2 = df2.iloc[:, :-1]
        y_tmp2 = df2.iloc[:, -1]
        X_train, X_test,y_train, y_test = train_test_split(X_tmp2, y_tmp2, test_size=0.2,random_state = args["rs"],shuffle=True,stratify=y_tmp2) 
        splitted_congifs.append([X_train, X_test,y_train, y_test])
        

# run the full process of cv, and test on the sets
for config in splitted_congifs:
    X_train, X_test,y_train, y_test = config[0],config[1],config[2],config[3] #8.1 - from ofir- here add stratified param on rate of responders
    
    lite_mode = True # use for debbuging only. using one small grid
    
    if lite_mode: # just for debugging. using one small grid
        param_pipe_list = [[param2,pipe2]]

    if not lite_mode: # full grid search , all models

        # pipe is represent the steps we want to execute, param represents which args we want to execute with
        param_pipe_list = [[param1a,pipe1a],[param1b,pipe1b],[param2,pipe2],[param3,pipe3],
        [param4,pipe4],[param5,pipe5],[param6,pipe6],[param7,pipe7]]

    # randomized_search = False
    for pair in param_pipe_list:
        yt,yp = [],[]
        param = pair[0]
        pipe = pair[1]
        search = RandomizedSearchCV(pipe,param,n_iter=args["n_iter"],cv =args["cv"],verbose=3 ,random_state = args['rs'],scoring = scorer(), refit=True).fit(X_train.astype(float),y_train)
        #search = RandomizedSearchCV(pipe,param,n_iter=args["n_iter"],cv =args["cv"],verbose=3 ,random_state = args['rs'],scoring = scorer(), refit=True).fit(X_train.astype(float),y_train.squeeze())
        cnt_splits = args['cv']
        best_ind = search.best_index_
        chooseThese = range(best_ind*cnt_splits,best_ind*cnt_splits+cnt_splits,1) # the exact range of the 5 test scores of the best index configuration
        yp_best = [yp[index] for index in chooseThese]
        #print("indexes range - folds of best configuration (chooseThese): ",chooseThese)
        yp_cv = np.concatenate(yp_best)
        yt_best = [yt[index] for index in chooseThese]
        yt_cv = np.concatenate(yt_best)    # print some more conclusions and details about the winning cv parmas and pipe and save them to csv          
        print_conclusions(X_train,pipe,search,yt_cv,yp_cv)
     


    
plt.show()
