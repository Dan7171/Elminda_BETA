import math
from functools import reduce

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
from sklearn.neural_network import MLPClassifier
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectKBest, f_classif, f_regression, mutual_info_classif
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from catboost import CatBoostClassifier
import warnings
import main_caller_r2
from sklearn.exceptions import DataConversionWarning
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, roc_auc_score, precision_score, recall_score
from sklearn.metrics import make_scorer
from sklearn.datasets import load_iris
from sklearn.utils import shuffle
import random
from sklearn.base import BaseEstimator, TransformerMixin
import os
from scipy.stats import randint
from runArguments import args
from imblearn.over_sampling import SMOTE
from sklearn.experimental import enable_halving_search_cv  # noqa
from sklearn.model_selection import HalvingRandomSearchCV
from imblearn.over_sampling import SMOTE
from sklearn.datasets import make_classification
import sys

warnings.filterwarnings("ignore", category=DeprecationWarning)
np.seterr(divide='ignore', invalid='ignore')
warnings.filterwarnings(action='ignore')
warnings.simplefilter(action='ignore')


def fxn():
    warnings.warn("deprecated", DeprecationWarning)


with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fxn()

all_splits_yts, all_splits_yps = [], []
my_scorer = args['scoring_method']


# def toy_data_run():
#
#     # *** TOY DATA PROCESSING ***:
#     # used that part of code to verify the pipeplines can scre well. they do. scores of randsearch cv's were around 0.95
#
#     iris = load_iris()
#     X_toy=iris.data
#     y_toy=iris.target
#     print(X_toy)
#     print(y_toy)
#     X_train, X_test,y_train, y_test = train_test_split(X_toy, y_toy, test_size=0.15,random_state = args["rs"])
#     rand_search_1a_iris = RandomizedSearchCV(pipe1a,param1a,n_iter=args["n_iter"],refit=True,cv =args["cv"],verbose=1,random_state=args["rs"]).fit(X_train,y_train.ravel())
#     print_conclusions("1a",rand_search_1a_iris)
#     rand_search_1b_iris = RandomizedSearchCV(pipe1b,param1b,n_iter=args["n_iter"],refit=True,cv =args["cv"],verbose=1,random_state=args["rs"]).fit(X_train,y_train.ravel())
#     print_conclusions("1b", rand_search_1b_iris)
#     rand_search_2_iris = RandomizedSearchCV(pipe2,param2,n_iter=args["n_iter"],refit=True,cv =args["cv"],verbose=1,random_state=args["rs"]).fit(X_train,y_train.ravel())
#     print_conclusions("2",rand_search_2_iris)
#     rand_search_3_iris = RandomizedSearchCV(pipe3,param3,n_iter=args["n_iter"],refit=True,cv =args["cv"],verbose=1,random_state=args["rs"]).fit(X_train,y_train.ravel())
#     print_conclusions("3",rand_search_3_iris)
#     rand_search_4_iris = RandomizedSearchCV(pipe4,param4,n_iter=args["n_iter"],refit=True,cv =args["cv"],verbose=1,random_state=args["rs"]).fit(X_train,y_train.ravel())
#     print_conclusions("4", rand_search_4_iris)
#     rand_search_5_iris = RandomizedSearchCV(pipe5,param5,n_iter=args["n_iter"],refit=True,cv =args["cv"],verbose=1,random_state=args["rs"]).fit(X_train,y_train.ravel())
#     print_conclusions("5", rand_search_5_iris)
#     rand_search_6_iris = RandomizedSearchCV(pipe6,param6,n_iter=args["n_iter"],refit=True,cv =args["cv"],verbose=1,random_state=args["rs"]).fit(X_train,y_train.ravel())
#     print_conclusions("6", rand_search_6_iris)

# to add as a transformer to the pipelines (dropping corellated features after kbest)
# class CorrelationDropper(BaseEstimator, TransformerMixin): # this one works, dont delete it
#     def __init__(self, threshold=0.9):
#         self.threshold = threshold
#
#     def fit(self, X, y=None):
#         # Compute the correlation matrix
#         if args["drop_out_correlated"]:
#             corr_matrix = np.corrcoef(X, rowvar=False)
#             # Select upper triangle of correlation matrix
#             upper = np.triu(np.ones(corr_matrix.shape), k=1)
#             correlated_features = np.where(upper * np.abs(corr_matrix) > self.threshold)
#
#             # Drop all but one of the correlated features
#             self.to_drop = []
#             for i, j in zip(*correlated_features):
#                 if i in self.to_drop:
#                     continue
#                 self.to_drop.append(j)
#
#         return self
#
#     def transform(self, X, y=None):
#         # Drop the correlated features
#         if args["drop_out_correlated"]:
#             return np.delete(X, self.to_drop, axis=1)
#         return X
#
# class CorrelationDropper2(BaseEstimator, TransformerMixin):
#     def __init__(self, threshold=0.9):
#         self.threshold = threshold
#         self.indices_to_keep = []
#
#     def _find_correlated_groups(self, correlations):
#         correlated_groups = [None]*correlations.shape[1]
#         visited_features = set()
#
#         for i in range(correlations.shape[0]):
#             marked = False
#             for j in range(i+1, correlations.shape[1]):
#                 if abs(correlations[i, j]) > self.threshold:
#                     marked = True
#                     # Add correlated features to the same group
#                     group = correlated_groups[i] if i in visited_features else correlated_groups[j] if j in visited_features else set()
#                     group.add(i)
#                     group.add(j)
#                     visited_features.add(i)
#                     visited_features.add(j)
#                     correlated_groups[i] = group
#                     correlated_groups[j] = group
#             if not marked:
#                 group = set()
#                 group.add(i)
#                 visited_features.add(i)
#                 correlated_groups[i] = group
#
#         print(correlated_groups)
#         return correlated_groups
#
#     def fit(self, X, y=None):
#         if args["drop_out_correlated"]:
#             # Calculate pairwise correlations between features
#             correlations = np.corrcoef(X, rowvar=False)
#             print(correlations)
#             # Identify correlated groups
#             correlated_groups = self._find_correlated_groups(correlations)
#
#             # Drop all correlated features but one from each group
#             self.indices_to_keep = []
#             for group in correlated_groups:
#                 if len(group) >= 1:
#                     self.indices_to_keep.append(list(group)[0])
#             print(self.indices_to_keep)
#         return self
#
#     def transform(self, X, y=None):
#         if args["drop_out_correlated"]:
#             # Select desired features from array
#             X_transformed = np.take(X, self.indices_to_keep, axis=1)
#             return X_transformed
#         return X
#

def generate_random_architectures(first_layer_size_options: tuple = (3, 5, 10, 20, 30), avg_num_of_layers_in_network=4,
                                  num_of_networks_to_create=100, adjacent_layers_ratio=1.33):
    """
    -creates num_of_networks_to_create network architectures (tuples of numbers,tuple[i]  represents size of ith hidden layer network)
    -avg number of hidden layers in each architecture will be avg_num_of_layers_in_network
    -first_layer_size_options is a tuple of optional values for the number of neurons in first layer of networks.
    -adjacent_layers_ratio - the i+1 layer size will be selected in fair chances to be one of three items: ( adjacent_layers_ratio * i layer size, (1/adjacent_layers_ratio)* i layer size, i layer size)
    - return a list of the architectures
    """
    # generate random layers
    # first_layer_size_options = (4,5,8,10,15,20,25,28,30,32,35,37,40)
    architectures = set()
    num_of_architectures = num_of_networks_to_create  # number of layer architertures to make
    for i in range(num_of_architectures):  # ith network to create
        architecture = []
        # avg_num_of_layers_in_network = 4 # set this num to the one you want
        for j in range(
                2 * avg_num_of_layers_in_network):  # jth layer in network (in expectancy, half of layers won't be crated)
            to_add_layer = random.random()  # throw a fair coin
            if to_add_layer < 0.5:
                continue  # don't add new layer
            # add new layer
            if len(architecture) == 0:
                cur_layer_options = first_layer_size_options  # first hidden layer size- pick number from all options
            else:
                cur_layer_options = [architecture[-1], min(math.floor(architecture[-1] * adjacent_layers_ratio), 50),
                                     max(math.floor(architecture[-1] * (1 / adjacent_layers_ratio)),
                                         1)]  # second layer or above layer size - pick something relatively close to prev layer

            # select randomly layer size
            index = random.randint(0, len(cur_layer_options) - 1)  # select a random index
            architecture.append(cur_layer_options[index])  # select layer size, and add it as next layer of architecute
        if len(architecture) > 0:
            architectures.add(tuple(architecture))
            print(f"network {i}\n {tuple(architecture)}")

    print(f"created {num_of_architectures} networks with avg size of layers {avg_num_of_layers_in_network}")
    return list(architectures)


def print_conclusions(df, pipe, search, best_cv_iter_yts_list_ndarray=None, best_cv_iter_yps_list_ndarray=None):
    print("-----------------------\n New CV report \n-----------------------")
    if args['classification']:
        name = pipe.named_steps['classifier']
        print("* Classifier: \n", name)
    else:
        name = pipe.named_steps['regressor']
        print("* Regressor: \n", name)
    print("* User arguments: \n", args)
    print("* Pipeline details: \n", str(pipe))
    print("* Best Hyperparametes picked in cross validation: (cv's best score): \n", search.best_params_)

    # features Kbest selected
    selected_features = ""
    if 'kBest' in pipe.named_steps:
        selector = search.best_estimator_.named_steps['kBest']
        selected_features = df.columns[selector.get_support()].tolist()
    print("* Best features by (selectKbest): \n", selected_features)

    # score 
    print("* Scorer_used:", args['scoring_method'])  # scoring method used for cv as scorer param
    score_mean = search.cv_results_['mean_test_score'][search.best_index_]
    score_std = search.cv_results_['std_test_score'][search.best_index_]
    print("* CV Score (cv's best score for best hyperparametes): %.3f +/- %.3f (see score func in hyperparams) " % (
        score_mean, score_std), "\n")

    cm = confusion_matrix(best_cv_iter_yts_list_ndarray, best_cv_iter_yps_list_ndarray)
    cm_with_legend = str(cm) + "\n" + "[[TN FP\n[FN TP]]"
    print("* Confusion matrix: \n", cm_with_legend)

    # confusion matrix plot making: 
    fig = metrics.ConfusionMatrixDisplay.from_predictions(best_cv_iter_yts_list_ndarray, best_cv_iter_yps_list_ndarray)

    fig.ax_.set_title(name)

    # calculate Response rate: 
    y_train_responders_cnt = 0

    if args['classification']:
        positive_label = 1  # can vary in different expirements
        for y_val in y_train.values:
            if y_val == positive_label:
                y_train_responders_cnt += 1

    else:  # regression
        for y_val in y_train.values:
            if y_val < -50:
                y_train_responders_cnt += 1

    print("* Response rate: ", y_train_responders_cnt / len(y_train))
    true_count = cm[0][0] + cm[1][1]
    false_count = cm[0][1] + cm[1][0]
    total = true_count + false_count
    precision = cm[1][1] / (cm[1][1] + cm[0][1])
    recall = cm[1][1] / (cm[1][1] + cm[1][0])
    accuracy = true_count / total
    f1 = 2 * (precision * recall) / (precision + recall)
    print("* CV Precision: ", precision)
    print("* CV Recall: ", recall)
    print("* CV Accuracy: ", accuracy)
    print("* CV F1: ", f1)

    # save cross val scores and params in tuning.csv for tracking
    # csv column names and values in a row:
    d = {
        "date": str(datetime.date.today()),  # date, hour
        "classifier": str(pipe.named_steps['classifier']),
        "pipe_named_steps": str(pipe.named_steps),  # all pipe steps, including the classifier as estimator
        "best_params": str(search.best_params_),
        "selected_features": str(selected_features),
        "user_inputs": str(args),  # runArguments.args
        "responders_rate": str(((cm[1][0] + cm[1][1]) / total)),  # responders / total
        "X_train_size(num of rows in cv input df)": str(total),
        "scorer_used": args['scoring_method'],  # scoring method used for cv as scorer param
        "scorer_score_mean": str(score_mean),  # should be one of (accuracy,precission,recall,f1 (if scorer_used = f1))
        "scorer_score_std": str(score_std),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": cm_with_legend,
        "param_grid_searched_at": str(param)
    }
    # save to cv:
    tmp = pd.DataFrame(d, index=[d.keys()])
    file_path = 'tuning.csv'
    if not os.path.exists(file_path):
        # If the file does not exist, create a new empty CSV file
        open(file_path, 'a').close()
        tmp.to_csv(file_path, index=False)
        return
    df = pd.read_csv(file_path)
    df = pd.concat([df, tmp])
    df.to_csv(file_path, index=False)

    print("CV report saved to  ", file_path)
    print("-----------------------\n End of CV report \n-----------------------", '\n' * 3)


def reset_column_to_original_labels(df, col, original_lables):
    df[col] = original_lables[df[col]]
    return df


def CV_Score(y_true, y_pred):
    """
    calculates score in specific split out of n_splits in specific iteration of n_iteratins in search

    :y_true: split's true values of y (validation values)
    :y_pred: y values in split as predicted by model
    """

    global my_scorer  # scorer user selected
    global all_splits_yts, all_splits_yps

    y_true = y_true[y_name].values  # change from df to ndarray for printing nicely

    all_splits_yts.append(y_true)
    all_splits_yps.append(y_pred)

    # input check
    if my_scorer not in ['accuracy', 'f1', 'roc_auc', 'precision', 'recall']:
        print('invalid score func from user')
        exit()
    cvscore = 0
    if my_scorer == 'roc_auc':
        cvscore = roc_auc_score(y_true, y_pred)
    elif my_scorer == 'accuracy':
        cvscore = accuracy_score(y_true, y_pred)
    elif my_scorer == 'f1':
        cvscore = f1_score(y_true, y_pred)
    elif my_scorer == 'precision':
        cvscore = precision_score(y_true, y_pred)
    elif my_scorer == 'recall':
        cvscore = recall_score(y_true, y_pred)

    if cvscore is None:
        print("problem - cvscore is nan.check y_true, y_pred...")
        exit()

    # y_true = y_true[y_name].values  # change from df to ndarray for printing nicely
    param_choice_idx = total_folds[0] // len(choice_scores)

    idx = total_folds[0] % len(choice_scores)  # num of folds elapsed % num of folds in each param choice
    choice_scores[idx] = cvscore
    max_param_choice_idx = args['n_iter'] - 1
    max_split_index = total_splits - 1

    # if idx == 0:
    #     param_choice_str = f"Parameter choice num {param_choice_idx} / {max_param_choice_idx} - starting..."
    #     print(param_choice_str)
    # if idx == len(choice_scores) - 1:  # calculated score for all folds in current parmeter coice
    #
    #     choice_avg_score = np.mean(choice_scores)
    #     choice_avg_score_str = f"In parameter choice num {param_choice_idx} / {max_param_choice_idx} avg score was: {choice_avg_score}. This is the best score so far"
    #
    #     improvement_report_str = None
    #     if choice_avg_score > best_score_by_now[0]:
    #         best_score_by_now[0] = choice_avg_score
    #         improvement_report_str = f"New best score is {best_score_by_now[0]}"
    #         # parmams_str = f"chosen parameters: {pipe.get_params()}" # param configuration chose
    #
    #     if improvement_report_str is not None:  # improvement in score
    #         print("New improvement!")
    #         print(improvement_report_str)
    #         # choice = search.cv_results_['params'][param_choice_idx]  # param grid of param choice which improved
    #         if not os.path.exists(search_statistics):
    #             open(search_statistics, 'w+').close()  # make file
    #         with open(search_statistics, "a+") as statistics:
    #             print(choice_avg_score_str)
    #             print(f"updating {search_statistics}...")
    #             statistics.write(f"{choice_avg_score_str}\n\n")
    #             print("statistics file updated successfully with new improvement in score message!")
    #     print(f"Best parameter choice score by now is {best_score_by_now[0]}")
    #     print(choice_avg_score_str)

    if idx == 0:
        param_choice_str = f"Parameter choice num {param_choice_idx} / {max_param_choice_idx} - starting..."
        print(param_choice_str)

    # max_split_index = total_splits - 1
    print(f"{total_folds[0]} / {max_split_index} splits counted in cross val search ")
    print("fold's true y \n", y_true)
    print("fold's predicted y\n", y_pred)
    print(f"scoring metric: {my_scorer}, score: {cvscore} ")


    if idx == len(choice_scores) - 1:  # calculated score for all folds in current parmeter coice
        choice_avg_score = np.mean(choice_scores)
        choice_avg_score_str = f"In parameter choice num {param_choice_idx} / {max_param_choice_idx} avg score was: {choice_avg_score}. This is the best score so far"

        improvement_report_str = None
        if choice_avg_score > best_score_by_now[0]:
            best_score_by_now[0] = choice_avg_score
            improvement_report_str = f"New best score is {best_score_by_now[0]}"
            # parmams_str = f"chosen parameters: {pipe.get_params()}" # param configuration chose

        if improvement_report_str is not None:  # improvement in score
            print("New improvement!")
            print(improvement_report_str)
            # choice = search.cv_results_['params'][param_choice_idx]  # param grid of param choice which improved
            if not os.path.exists(search_statistics):
                open(search_statistics, 'w+').close()  # make file
            with open(search_statistics, "a+") as statistics:
                print(choice_avg_score_str)
                print(f"updating {search_statistics}...")
                statistics.write(f"{choice_avg_score_str}\n\n")
                print("statistics file updated successfully with new improvement in score message!")
        print(f"Best parameter choice score by now is {best_score_by_now[0]}")
        print(choice_avg_score_str)
    total_folds[0] += 1  # splits counter, starts from 0

    return cvscore


def scorer():
    return make_scorer(CV_Score)


# ===================================================


# *****************************************************************************************************
# ******************************************* MAIN ****************************************************
# *****************************************************************************************************


# write output to logfile (to ease hyper - parameter tuning)
print(" >>>>>>>>>>>>>>>>>>>>> STARTING MAIN OF GSCVrunner.py >>>>>>>>>>>>>>>>>>>>>")

if args['stdout_to_file']:
    logfile_name = 'stdout.txt'
    if os.path.exists(logfile_name):
        os.remove(logfile_name)
    log_file = open(logfile_name, 'a')
    sys.stdout = sys.stderr = log_file

search_statistics = "search_statistics.txt"

if os.path.exists(search_statistics):
    os.remove(search_statistics)
print(args)
if args['classification']:
    y_name = '6-weeks_HDRS21_class'  # classification problem (prediciting change rate class)
else:
    y_name = "6-weeks_HDRS21_change_rate"  # regression problem

X, y = main_caller_r2.get_X_y(y_name, args["X_version"])  # X and y's creationa and processing

X.reset_index(inplace=True, drop=True)
if args['classification']:
    y[y_name] = y[y_name].replace({"responsive": 1, "non_responsive": 0})

if args["age_under_50"]:  # using only candidated under age of 50
    df = X.join(y)
    df = df[df['age'] < 50]
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1:]

# create the piplelines and greeds:

# models to use in pipelines

# Principal component analysis (PCA)
pca = PCA()

# Standardize features by removing the mean and scaling to unit variance.
# The standard score of a sample x is calculated as:
# z = (x - u) / s
scaler = StandardScaler()

# Select features according to the k highest scores
kBest_selector = SelectKBest()

##################### CLASSIFICATION ##############################

# classifiers (estimators for the piplene-final step)
if args['classification']:
    clf1 = LogisticRegression(random_state=args["rs"])
    clf2 = KNeighborsClassifier()
    clf3 = SVC(probability=True, random_state=args["rs"])
    clf4 = DecisionTreeClassifier(random_state=args["rs"])
    clf5 = RandomForestClassifier(random_state=args["rs"])
    clf6 = GradientBoostingClassifier(random_state=args["rs"])
    clf7 = CatBoostClassifier(random_state=args["rs"], logging_level='Silent')
    clf8 = MLPClassifier(random_state=args["rs"])
    # The param 'grids'
    # note: parameters of different models pipelines can be set using '__' separated parameter names. modelname__parameter name = options to try ing gscv:

    param1a = {  # LOGISTIC REGRESSION + pca
        "pca__n_components": range(2, 50, 3),
        "classifier__C": [0.001, 0.01, 0.1, 1, 10, 100],
        "classifier__penalty": ['l2'],
        "classifier": [clf1]
    }
    # tune 3. c=0.5, k = 16, penalty= l2,mutual_info
    param1b = {  # LOGISTIC REGRESSION + kbest,
        "classifier__C": [0.45, 0.47, 0.5, 0.52, 0.55, 0.6],  # classifier (logistic regression) param 'C' for tuning
        "kBest__k": range(15, 17),  # selctKbest param 'k'for tuning. must be  <= num of features
        'classifier__penalty': ['l2'],
        "kBest__score_func": [mutual_info_classif],  # selctKbest param 'score_func'for tuning
        "classifier": [clf1]

    }
    param2a = {  # knn + pca
        "pca__n_components": [i for i in range(3, 100, 5)],
        "classifier__n_neighbors": range(1, 90, 3),
        "classifier__weights": ['uniform', 'distance'],
        "classifier__algorithm": ['auto', 'ball_tree', 'kd_tree', 'brute'],
        "classifier__leaf_size": range(3, 80, 3),
        "classifier__p": [2],
        "classifier": [clf2]
    }

    param2b = {  # KNN + kbest
        "classifier__n_neighbors": range(1, 90, 3),
        "classifier__weights": ['uniform', 'distance'],
        "classifier__algorithm": ['auto', 'ball_tree', 'kd_tree', 'brute'],
        "classifier__leaf_size": range(3, 80, 3),
        "classifier__p": [2],
        "kBest__k": range(4, 100, 3),
        "kBest__score_func": [f_classif, mutual_info_classif],  # selctKbest param 'score_func'for tuning
        "classifier": [clf2]
    }
    param3a = {  # SVC + pca
        "pca__n_components": [i for i in range(3, 100, 5)],
        'classifier__gamma': [60, 65, 70, 75, 80],
        'classifier__kernel': ['linear', 'rbf', 'sigmoid'],
        'classifier__C': [900, 950, 1000, 1100, 1200, 1400],
        "classifier": [clf3]
    }
    param3b = {  # SVC + kbest
        "kBest__k": range(40, 70, 5),  # k should be smaller than num of features in X
        'classifier__gamma': [60, 65, 70, 75, 80],
        'classifier__kernel': ['linear', 'rbf', 'sigmoid'],
        'classifier__C': [900, 950, 1000, 1100, 1200, 1400],
        "classifier": [clf3]
    }
    param4a = {  # DECISION TREE + pca
        #DecisionTreeClassifier(random_state=42),
        # classifier__criterion=entropy, classifier__max_depth=40, classifier__max_leaf_nodes=35,
        # classifier__min_samples_leaf=5, classifier__min_samples_split=14, pca__n_components=53;,
        # score=0.704 total time=   0.1s

        # best accuracy - 0.714021164021164
        # {'pca__n_components': 52, 'classifier__min_samples_split': 8,
        # 'classifier__min_samples_leaf': 3, 'classifier__max_leaf_nodes': 95,
        # 'classifier__max_depth': 240, 'classifier__criterion': 'entropy',
        # 'classifier': DecisionTreeClassifier(criterion='entropy', max_depth=240, max_leaf_nodes=95,
        #                        min_samples_leaf=3, min_samples_split=8,
        #                        random_state=42)}

        # "{'pca__n_components': 50, 'classifier__min_samples_split': 2,
        # 'classifier__min_samples_leaf': 2, 'classifier__max_leaf_nodes': 80, 'classifier__max_depth': 200,
        # 'classifier__criterion': 'entropy', 'classifier': DecisionTreeClassifier(criterion='entropy', max_depth=200, max_leaf_nodes=80,
        #                        min_samples_leaf=2, random_state=42)}"

        # "{'pca__n_components': 52, 'classifier__min_samples_split': 2,
        # 'classifier__min_samples_leaf': 2, 'classifier__max_leaf_nodes': 97,
        # 'classifier__max_depth': 239, 'classifier__criterion': 'entropy',
        # 'classifier': DecisionTreeClassifier(criterion='entropy', max_depth=239, max_leaf_nodes=97,
        #                        min_samples_leaf=2, random_state=42)}"

        #"{'pca__n_components': 52, 'classifier__min_samples_split': 3, 'classifier__min_samples_leaf': 2,
        # 'classifier__max_leaf_nodes': 95, 'classifier__max_depth': 240, 'classifier__criterion': 'entropy',
        # 'classifier': DecisionTreeClassifier(criterion='entropy', max_depth=240, max_leaf_nodes=95,
        #               min_samples_leaf=2, min_samples_split=3,
        #               random_state=42)}"

        "pca__n_components": range(48,55),
        'classifier__max_leaf_nodes': range(93,102),
        'classifier__max_depth': range(236,249),
        'classifier__criterion': ['entropy'],
        'classifier__min_samples_split': range(2,4),
        'classifier__min_samples_leaf': range(2,4),
        "classifier": [clf4]
    }

    # DECISION TREE + kbest
    param4b = {
        'classifier__max_leaf_nodes': range(1, 25, 3),
        'classifier__max_depth': [2, 4, 6, 8, 10, 12],
        'classifier__criterion': ['gini', 'entropy'],
        'classifier__min_samples_split': range(2, 40, 5),
        # reason I tried this classifier params https://medium.com/analytics-vidhya/decisiontree-classifier-working-on-moons-dataset-using-gridsearchcv-to-find-best-hyperparameters-ede24a06b489
        "classifier": [clf4]
        }


    param5a = {  # RANDOM FOREST + pca

        # # best by now (accuracy = 0.756) (cv=5)

        # {'pca__n_components': 76, 'classifier__min_samples_split': 12, 'classifier__min_samples_leaf': 10,
        #  'classifier__max_features': 2, 'classifier__max_depth': 43, 'classifier__bootstrap': True,
        #  'classifier': RandomForestClassifier(max_depth=43, max_features=2, min_samples_leaf=10,
        #                                       min_samples_split=12, random_state=42)}

        # second best: (accuracy = 0.72) (cv=4)

        # {'pca__n_components': 60, 'classifier__min_samples_split': 3, 'classifier__min_samples_leaf': 3,
        #  'classifier__max_features': 'auto', 'classifier__max_depth': 60, 'classifier__bootstrap': True,
        #  'classifier': RandomForestClassifier(max_depth=60, max_features='auto', min_samples_leaf=3,
        #                                       min_samples_split=3, random_state=42)}

        # third best (accuracy = 0.75) (cv=5)
        # {'pca__n_components': 71, 'classifier__min_samples_split': 7,
        # 'classifier__min_samples_leaf': 5, 'classifier__max_features': 2,
        # 'classifier__max_depth': 75, 'classifier__bootstrap': True, 'classifier':
        # RandomForestClassifier(max_depth=75, max_features=2, min_samples_leaf=5,
        #                        min_samples_split=7, random_state=42)}
        "pca__n_components": [i for i in range(55, 85,2)],
        'classifier__bootstrap': [True],
        "classifier__max_depth": [35,75,2], #11 evenly spaceced num in range10-110
        "classifier__min_samples_split": range(2, 15),
        "classifier__min_samples_leaf": range(2, 12),
        "classifier__max_features": ['auto', 'sqrt',2],
        "classifier": [clf5]
    }
    param5b = {  # RANDOM FOREST + kbest
        # reason I tried this classifier params:
        # {'kBest__k': 364, 'classifier__min_samples_split': 2,
        # 'classifier__min_samples_leaf': 2, 'classifier__max_features': 'sqrt',
        # 'classifier__max_depth': 35, 'classifier__bootstrap': True, 'classifier':
        # RandomForestClassifier(max_depth=35, min_samples_leaf=2, random_state=42)}

        # * CV Accuracy:  0.6985294117647058:
        # * Best Hyperparametes picked in cross validation: (cv's best score):
        #  {'kBest__k': 340, 'classifier__min_samples_split': 2, 'classifier__min_samples_leaf': 2,
        #  'classifier__max_features': 'auto', 'classifier__max_depth': 12, 'classifier__bootstrap': True, 'classifier':
        #  RandomForestClassifier(max_depth=12, max_features='auto', min_samples_leaf=2,

        # #
        # {'kBest__k': 485, 'classifier__min_samples_split': 6, 'classifier__min_samples_leaf': 2,
        #  'classifier__max_features': 'auto', 'classifier__max_depth': 154, 'classifier__bootstrap': False,
        #  'classifier': RandomForestClassifier(bootstrap=False, max_depth=154, max_features='auto',
        #                                       min_samples_leaf=2, min_samples_split=6,
        #                                       random_state=42)}
        #
        #

        # {'kBest__k': 485, 'classifier__min_samples_split': 5, 'classifier__min_samples_leaf': 2,
        #  'classifier__max_features': 'auto', 'classifier__max_depth': 156, 'classifier__bootstrap': False,
        #  'classifier': RandomForestClassifier(bootstrap=False, max_depth=156, max_features='auto',
        #                                       min_samples_leaf=2, min_samples_split=5,
        #                                       random_state=42)}
        #
        "kBest__k": range(480, 490),
        'classifier__bootstrap': [True,False],
        "classifier__max_depth": range(150,160),
        "classifier__min_samples_split": range(2,10),
        "classifier__min_samples_leaf": range(2,10),
        "classifier__max_features": ['auto','sqrt'],
        "classifier": [clf5]
    }

    param6a = {  # GRADIENT BOOSTING + pca
        # best - accuracy 0.78
        # {'pca__n_components': 83, 'classifier__subsample': 0.9,
        # 'classifier__n_estimators': 40, 'classifier__min_samples_split': 2,
        # 'classifier__min_samples_leaf': 1, 'classifier__max_features': None,
        # 'classifier__max_depth': 5, 'classifier__learning_rate': 0.01,
        # 'classifier': GradientBoostingClassifier(learning_rate=0.01,
        # max_depth=5, n_estimators=40,
        #  random_state=41, subsample=0.9)}
        "pca__n_components": [i for i in range(70, 100, 2)],
        'classifier__n_estimators': [i for i in range(20,60,3)],
        'classifier__learning_rate': [0.0001,0.01],
        'classifier__max_depth': [2,3,4,5,6, 7,8],
        'classifier__min_samples_split': [1,2,3],
        'classifier__min_samples_leaf': [1,2,3],
        'classifier__max_features': ['auto', 'sqrt', None],
        'classifier__subsample': [0.8, 0.9, 1],
        "classifier": [clf6]
    }

    param6b = {  # GRADIENT BOOSTING + kbest
        # reason I tried this classifier params:
        "kBest__k": range(2, 600, 20),
        'classifier__subsample': [0.6, 0.9],
        'classifier__n_estimators': [30,35,40,45,50,60,65,70],
        'classifier__min_samples_split': [1,2,3,4,5],
        'classifier__min_samples_leaf': [1,2,3,4,5],
        'classifier__max_features': ['auto',None,'sqrt'],
        'classifier__max_depth': [2,4,7,12],
        'classifier__learning_rate': [0.01],
        "classifier": [clf6]
    }
    # GRADIENT BOOSTING ( NO PCA  NO KBEST)
    param6c = {
        # 'classifier__n_estimators': [5, 20, 35, 50, 65, 80, 95, 110, 125, 150, 200, 250, 350],
        'classifier__learning_rate': [0.0001],
        'classifier__max_depth': [2, 15, 40,60,80,100,200,500],
        'classifier__min_samples_split': [1,2,9,16,24,38],
        'classifier__min_samples_leaf': [1,2,9,16,24,38],
        'classifier__max_features': ['auto', 'sqrt'],
        'classifier__subsample': [0.6, 0.95],
        "classifier": [clf6]
    }

    param7a = {  # CATBOOST CLASSIFIER + pca
        "pca__n_components": [i for i in range(3, 100, 5)],
        'classifier__n_estimators': [3, 10, 30, 50, 100, 500],
        "classifier__learning_rate": [0.0001, 0.01, 0.1],
        'classifier__subsample': [0.1, 0.3, 0.5, 0.7, 1.0],
        'classifier__max_depth': range(3, 40, 3),
        "classifier": [clf7]
    }
    param7b = {  # CATBOOST CLASSIFIER + kbest
        "kBest__k": range(4, 40, 8),
        'classifier__n_estimators': [3, 10, 30, 50, 100, 500],
        "classifier__learning_rate": [0.0001, 0.01, 0.1],
        'classifier__subsample': [0.1, 0.3, 0.5, 0.7, 1.0],
        'classifier__max_depth': range(3, 40, 3),
        "classifier": [clf7]
    }

    param8a = {  # MLPClassifier (neural network) + PCA
        #best by now: (accuracy 0.82)
        # {'pca__n_components': 26, 'classifier__verbose': False, 'classifier__solver': 'adam',
        #  'classifier__max_iter': 2500, 'classifier__learning_rate': 'adaptive',
        #  'classifier__hidden_layer_sizes': (27, 31, 21, 31), 'classifier__alpha': 0.0001,
        #  'classifier__activation': 'relu',
        #  'classifier': MLPClassifier(hidden_layer_sizes=(27, 31, 21, 31), learning_rate='adaptive',
        #                              max_iter=2500, random_state=42)}
        "pca__n_components": [i for i in range(24, 29)],
        'classifier__hidden_layer_sizes': [(i, j, k, l) for i in range(25, 30) for j in range(29, 34) for k in
                                           range(19, 24)
                                           for l in range(29, 34)],
        'classifier__activation': ['relu'],
        'classifier__solver': ['adam'],
        'classifier__alpha': [0.0001],
        'classifier__learning_rate': ['adaptive'],
        'classifier__max_iter': [2800],
        'classifier__verbose': [False],  # details prints of loss
        "classifier": [clf8]
    }
    param8b = {  # MLPClassifier (neural network) + KBEST
        "kBest__k": range(26,30),
        'classifier__hidden_layer_sizes': [(i, j, k, l) for i in range(25, 30) for j in range(29, 34) for k in
                                           range(19, 24)
                                           for l in range(29, 34)],
        'classifier__activation': ['identity', 'logistic', 'tanh', 'relu'],
        'classifier__solver': ['adam','lbfgs','sgd'],
        'classifier__alpha': [0.0001, 0.01],
        'classifier__learning_rate': ['adaptive','constant','invscaling'],
        'classifier__max_iter': [3500],
        'classifier__verbose': [False],  # details prints of loss
        "classifier": [clf8]

    }

    # define the pipelines

    # pipe a - kbest
    # pipe b - pca

    pipe1a = Pipeline(steps=[("scaler", scaler), ("pca", pca), ("classifier", param1a["classifier"][0])])
    pipe1b = Pipeline(steps=[("scaler", scaler), ("kBest", kBest_selector), ("classifier", param1b["classifier"][0])])
    pipe2a = Pipeline(steps=[("scaler", scaler), ("pca", pca), ("classifier", param2a["classifier"][0])])
    pipe2b = Pipeline(steps=[("scaler", scaler), ("kBest", kBest_selector), ("classifier", param2b["classifier"][0])])
    pipe3a = Pipeline(steps=[("scaler", scaler), ("pca", pca), ("classifier", param3a["classifier"][0])])
    pipe3b = Pipeline(steps=[("scaler", scaler), ("kBest", kBest_selector), ("classifier", param3b["classifier"][0])])
    pipe4a = Pipeline(steps=[("scaler", scaler), ("pca", pca), ("classifier", param4a["classifier"][0])])
    pipe4b = Pipeline(steps=[("scaler", scaler), ("kBest", kBest_selector), ("classifier", param4b["classifier"][0])])
    pipe5a = Pipeline(steps=[("scaler", scaler), ("pca", pca), ("classifier", param5a["classifier"][0])])
    pipe5b = Pipeline(steps=[("scaler", scaler), ("kBest", kBest_selector), ("classifier", param5b["classifier"][0])])
    # pipe5c = Pipeline(steps=[("scaler", scaler), ("classifier", param5c["classifier"][0])])
    pipe6a = Pipeline(steps=[("scaler", scaler), ("pca", pca), ("classifier", param6a["classifier"][0])])
    pipe6b = Pipeline(steps=[("scaler", scaler), ("kBest", kBest_selector), ("classifier", param6b["classifier"][0])])
    pipe6c = Pipeline(steps=[("scaler", scaler), ("classifier", param6c["classifier"][0])])
    pipe7a = Pipeline(steps=[("scaler", scaler), ("pca", pca), ("classifier", param7a["classifier"][0])])
    pipe7b = Pipeline(steps=[("scaler", scaler), ("kBest", kBest_selector), ("classifier", param7b["classifier"][0])])
    # pipe7c = Pipeline(steps=[("scaler", scaler), ("classifier", param7c["classifier"][0])])
    pipe8a = Pipeline(steps=[("scaler", scaler), ("pca", pca), ("classifier", param8a["classifier"][0])])
    pipe8b = Pipeline(steps=[("scaler", scaler), ("kBest", kBest_selector), ("classifier", param8b["classifier"][0])])
    # pipe8c = Pipeline(steps=[("scaler", scaler), ("classifier", param8c["classifier"][0])])
########################## regression ####################################

else:  # regression

    # neural network regression
    regressor8 = MLPRegressor(random_state=args['rs'])
    param8_reg = {
        'regressor__hidden_layer_sizes': [(10,), (20, 10)],
        'regressor__activation': ['relu', 'tanh', 'logistic'],
        'regressor__solver': ['adam', 'sgd'],
        'regressor__alpha': [0.0001, 0.001, 0.01],
        'pca__n_components': [2, 4, 6],
        'scaler': [None, scaler],
        'regressor': [regressor8]
    }
    pipe8_reg = Pipeline(steps=[("scaler", scaler), ("pca", pca), ("regressor", param8_reg["regressor"][0])])


#############
# balance number of responders and non responders  by undersampling the majority class (responders):

def balance_y_values(X, y, method):
    """
    making data set balanced in responders and non responders
    """

    if not args['classification']:
        print("balancing y values supported only in calassification mode")
        print("change args['classification'] to True and try over ")
        exit()

    print(f"balancing dataset (y values) using {method}...")
    print(f"{X.shape[0]} y values\nvalue counts:\n{y.value_counts()}")

    # select the method of balancing you want to use
    if method == "undersample_majority":  # drop values from the larger category

        data = X.join(y)
        # print("balancing y values...")
        # print(f"before balancing: {X.shape[0]} y values\nvalue counts:\n{y.value_counts()}")

        # undersample responders
        data = data.drop(data[data[y_name] == 1].sample(
            frac=.3).index)  # LEAVE frac=.3 (matching args['both'] = True and args['both]= False)  drop a fraction of the responders to equalize num of responders and non responders
        X = data.iloc[:, :-1]
        y = data.iloc[:, -1]

        return X, y

    elif method == 'SMOTE':  # generate fake observations from the minority category

        # Resampling the minority class. The strategy can be changed as required.
        sm = SMOTE(sampling_strategy='minority', random_state=42)
        # Fit the model to generate the data.
        X, y = sm.fit_resample(X, y)
        # oversampled = pd.concat([pd.DataFrame(oversampled_Y), pd.DataFrame(oversampled_X)], axis=1)

    print(f"after balancing: {X.shape[0]} y values\nvalue counts:\n{y.value_counts()}")
    return X, y


############


splitted_congifs = []  # each list is a list of X_train, X_test,y_train, y_test to run cv and fit on

# Split data by rows into categories (or not):
if (args["split_rows"] == 'normal'):  # regular test train split  =  don't drop subjects:
    if args['classification']:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args['test_size'], random_state=args["rs"],
                                                            shuffle=True, stratify=y)
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args['test_size'], random_state=args["rs"])
    splitted_congifs.append([X_train, X_test, y_train, y_test])

elif args["split_rows"] in ['h1', 'h7', 'h1h7']:  # split by 'Treatment_group' (device - h1/h7)
    print("splitting by h1 h7 : ")

    if args['split_rows'] in ['h1', 'h1h7']:  # use h1
        # for treatment group 1 - h1 ('Treatment_group' is 0)
        # h1 subjects only
        df1 = X.join(y)
        df1 = df1[df1['Treatment_group'] == 0]
        print("new data- only the rows where column 'Treatment_group is 0:")
        X_tmp = df1.iloc[:, :-1]
        y_tmp = df1.iloc[:, -1]
        # X_train, X_test,y_train, y_test = train_test_split(X_tmp, y_tmp, test_size=args['test_size'],random_state = args["rs"],shuffle=True,stratify=y_tmp) 
        if args['classification']:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args['test_size'],
                                                                random_state=args["rs"], shuffle=True, stratify=y_tmp)
        else:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args['test_size'],
                                                                random_state=args["rs"], shuffle=True)
        splitted_congifs.append([X_train, X_test, y_train, y_test])

    if args['split_rows'] in ['h7', 'h1h7']:  # use h7
        # for treatment group 2 - h7 ('Treatment_group' is 1) 
        # h7 subjects only
        # df2 = X[X['Treatment_group'] == 1].join(y, how = "inner") #H7
        df2 = X.join(y)
        df2 = df2[df2['Treatment_group'] == 1]
        print("new data- only the rows where column 'Treatment_group is 1:")
        X_tmp2 = df2.iloc[:, :-1]
        y_tmp2 = df2.iloc[:, -1]

        if args['classification']:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args['test_size'],
                                                                random_state=args["rs"], shuffle=True, stratify=y_tmp2)
        else:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args['test_size'],
                                                                random_state=args["rs"])

        splitted_congifs.append([X_train, X_test, y_train, y_test])

# run the full process of cv, and test on the sets
for config in splitted_congifs:
    print("configs types debug")
    print([type(config[i]) for i in range(4)])
    for i in range(4):
        if isinstance(config[i], pd.Series):  # convert to data frame to make scorer work
            config[i] = config[i].to_frame()

    X_train, X_test, y_train, y_test = config[0], config[1], config[2], config[
        3]  # 8.1 - from ofir- here add stratified param on rate of responders

    # fix imbalance in train set
    if args['balance_y_values']:
        X_train, y_train = balance_y_values(X_train, y_train, method='SMOTE')
# *******************************
    if args['classification']:
        if args['lite_mode']:  # just for debugging. using one small grid
            param_pipe_list = [[param4a, pipe4a]]
# ********************************
        else:  # more than one model
            # pipe is represent the steps we want to execute, param represents which args we want to execute with
            param_pipe_list = []  # put all the pipe and param paris you wabnt

    else:  # regression
        if args['lite_mode']:  # just for debugging. using one small grid
            param_pipe_list = [[param8_reg, pipe8_reg]]
    # randomized_search = False
    for pair in param_pipe_list:

        total_folds = [0]  # counter
        all_splits_yts, all_splits_yps = [], []  # all_splits_yts and all_splits_yps are lists of all all_splits_yts and all_splits_yps vectors, one vector for each split (n_iter * n_splits len).
        param = pair[0]
        pipe = pair[1]
        choice_scores = [0 for i in range(args['cv'])]  # list of scores in length num of folds
        best_score_by_now = [0]

        # cross validation search
        print(param)
        print(pipe)


        if args['exhaustive_grid_search']: # exhausitve search
            print("~~~~~~~~~~ EXHAUSTIVE SEARCH CV ~~~~~~~~~~~")
            search = GridSearchCV(param_grid=param,estimator=pipe,cv=args["cv"],verbose=3,scoring=scorer(),refit=True)
            choice_options = [len(val) for val in param.values()]
            args['n_iter'] = reduce(lambda x,y:x*y, choice_options) #multiply all choice options
            search.fit(X_train.astype(float), y_train)
        else: # randomized search
            print("~~~~~~~~~~ RANDOMIZED SEARCH CV ~~~~~~~~~~")
            search = RandomizedSearchCV(estimator=pipe, param_distributions=param, n_iter=args["n_iter"], cv=args["cv"], n_jobs=args['n_jobs'],verbose=3, random_state=args['rs'], scoring=scorer(), refit=True)

        n_splits = args['cv']  # num of splits in cv_iter (cv parameter n_splits)
        total_splits = args["n_iter"] * n_splits  # num of iterations in search * num of folds
        search.fit(X_train.astype(float), y_train)

        best_cv_iter_idx = search.best_index_  # index of the iteration in cross val search which had best parsms (0<=best_cv_iter_idx <=niter)
        best_cv_iter_first_split_idx = best_cv_iter_idx * n_splits
        best_cv_iter_all_splits_indices = range(best_cv_iter_first_split_idx,
                                                best_cv_iter_first_split_idx + n_splits,
                                                1)  # the exact range of the 5 test scores of the best index configuration

        # best_cv_iter_yps_list is a list in length (n_splits (n folds)) contains vectors from the best iteration
        # in cv
        best_cv_iter_yps_list = [all_splits_yps[i] for i in best_cv_iter_all_splits_indices]

        # print("indexes range - folds of best configuration (best_cv_iter_splits_indices): ",
        # best_cv_iter_splits_indices)
        best_cv_iter_yps_list_ndarray = np.concatenate(best_cv_iter_yps_list)  # turn to nd_array

        # best_cv_iter_yts_list is a list in length (n_splits (n folds)) contains vectors from the best iteration
        # in cv
        best_cv_iter_yts_list = [all_splits_yts[index] for index in best_cv_iter_all_splits_indices]
        best_cv_iter_yts_list_ndarray = np.concatenate(
            best_cv_iter_yts_list)  # print some more conclusions and details about the winning cv parmas and pipe and save them to csv
        print_conclusions(X_train, pipe, search, best_cv_iter_yts_list_ndarray, best_cv_iter_yps_list_ndarray)

print(f"<<<<<<<<<<<<<<<<<<<<< GSCVrunner.py finished successfuly, check {logfile_name}, {search_statistics} and "
      f"tuning.csv for search results <<<<<<<<<<<<<<<<<<<<<")
log_file.close()
plt.show()
