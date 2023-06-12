import math
from functools import reduce

from matplotlib import pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.svm import SVC
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import classification_report
import datetime
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectKBest, f_classif, f_regression, mutual_info_classif
from sklearn.pipeline import Pipeline
from sklearn.metrics import fbeta_score
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
from sklearn.experimental import enable_halving_search_cv  # noqa
from sklearn.model_selection import HalvingRandomSearchCV
from imblearn.over_sampling import SMOTE
from sklearn.datasets import make_classification
from imblearn.pipeline import Pipeline as imb_Pipeline
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


def print_conclusions(df, pipe, search, best_cv_iter_yts_list_ndarray=None, best_cv_iter_yps_list_ndarray=None,folder="out_folder"):
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
        "scorer_used": args['scoring_method'],
        # scoring method used for cv as scorer param should be one of (accuracy,precision,recall,f1
        "scorer_score_mean": str(score_mean),
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
    # file_path = 'tuning.csv'
    file_path  = os.path.join(folder,'tuning.csv')
    if not os.path.exists(file_path):
        # If the file does not exist, create a new empty CSV file
        open(file_path, 'a').close()
        tmp.to_csv(file_path, index=False)
        return
    df = pd.read_csv(file_path)
    df = pd.concat([df, tmp])
    df.to_csv(file_path, index=False)

    print("Try this as a ways to print report :")
    print(classification_report(y_true=best_cv_iter_yts_list,y_pred=best_cv_iter_yps_list ))
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
    if my_scorer not in ['accuracy', 'f1', 'roc_auc', 'precision', 'recall', 'f_beta']:
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
    elif my_scorer == 'f_beta':
        cvscore = fbeta_score(y_true, y_pred,
                              beta=0.5)  # beta = weight given to recall (weight given to precision is always 1)

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
now = datetime.datetime.now()
folder = now.strftime("%Y-%m-%d %H_%M_%S")
if not os.path.isdir(folder):
    os.mkdir(folder)
if args['stdout_to_file']:

    # logfile_name = 'stdout.txt'
    logfile_name = os.path.join(folder,'stdout.txt')
    print(f"see stdout in {logfile_name}")
    if os.path.exists(logfile_name):
        os.remove(logfile_name)
    log_file = open(logfile_name, 'a')
    sys.stdout = sys.stderr = log_file

# search_statistics = "search_statistics.txt"
search_statistics = os.path.join(folder,"search_statistics.txt")
if os.path.exists(search_statistics):
    os.remove(search_statistics)
# print(args)
if args['classification']:
    y_name = '6-weeks_HDRS21_class'  # classification problem (prediciting change rate class)
else:
    y_name = "6-weeks_HDRS21_change_rate"  # regression problem

X, y = main_caller_r2.get_X_y(y_name, args["X_version"])  # X and y's creationa and processing


if not os.path.exists(os.path.join(folder, 'X.csv')):
    X.to_csv(os.path.join(folder, 'X.csv'))
if not os.path.exists(os.path.join(folder, 'y.csv')):
    y.to_csv(os.path.join(folder, 'y.csv'))

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
        "pca__n_components": range(2, 500, 10),
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
        "pca__n_components": range(2, 55, 10),
        'classifier__gamma': range(20, 400, 10),
        'classifier__kernel': ['linear', 'rbf', 'sigmoid'],
        'classifier__C': range(20, 1000, 30),
        "classifier": [clf3]
    }
    param3b = {  # SVC + kbest
        "kBest__k": range(2, 55, 10),  # k should be smaller than num of features in X
        'classifier__gamma': range(5, 1000, 30),
        'classifier__kernel': ['linear', 'rbf', 'sigmoid'],
        'classifier__C': range(5, 1500, 20),
        "classifier": [clf3]
    }
    param4a = {  # DECISION TREE + pca

        "pca__n_components": range(2, 55, 10),
        'classifier__max_leaf_nodes': range(5, 1000, 30),
        'classifier__max_depth': range(5, 1000, 30),
        'classifier__criterion': ['entropy', 'gini'],
        'classifier__min_samples_split': range(2, 100, 5),
        'classifier__min_samples_leaf': range(20, 100, 5),
        "classifier": [clf4]
    }

    # DECISION TREE + kbest
    #
    # {'kBest__k': 475, 'classifier__min_samples_split': 3, 'classifier__min_samples_leaf': 2,
    #  'classifier__max_leaf_nodes': 80, 'classifier__max_depth': 260, 'classifier__criterion': 'gini',
    #  'classifier': DecisionTreeClassifier(max_depth=260, max_leaf_nodes=80, min_samples_leaf=2,
    #                                       min_samples_split=3, random_state=42)}

    # {'kBest__k': 500, 'classifier__min_samples_split': 3, 'classifier__min_samples_leaf': 2,
    # 'classifier__max_leaf_nodes': 86, 'classifier__max_depth': 274, 'classifier__criterion': 'gini',
    # 'classifier': DecisionTreeClassifier(max_depth=274, max_leaf_nodes=86, min_samples_leaf=2,
    #                   min_samples_split=3, random_state=42)}
    param4b = {
        "kBest__k": range(470, 800, 10),
        'classifier__max_leaf_nodes': range(78, 90, 2),
        'classifier__max_depth': range(260, 280, 3),
        'classifier__criterion': ['gini'],
        'classifier__min_samples_split': [2, 3],
        'classifier__min_samples_leaf': [1, 2],
        "classifier": [clf4]

    }

    param5a = {  # RANDOM FOREST + pca

        "pca__n_components": range(2, 40, 10),
        'classifier__bootstrap': [True, False],
        "classifier__max_depth": range(5, 1000, 30),  # 11 evenly spaceced num in range10-110
        "classifier__min_samples_split": range(2, 100, 5),
        "classifier__min_samples_leaf": range(2, 100, 5),
        "classifier__max_features": ['auto', 'sqrt', 2],
        "classifier": [clf5]
    }
    param5b = {  # RANDOM FOREST + kbest

        "kBest__k": range(2, 40, 10),
        'classifier__bootstrap': [True, False],
        "classifier__max_depth": range(150, 160),
        "classifier__min_samples_split": range(2, 10),
        "classifier__min_samples_leaf": range(2, 10),
        "classifier__max_features": ['auto', 'sqrt'],
        "classifier": [clf5]
    }

    param6a = {  # GRADIENT BOOSTING + pca
        #{'pca__n_components': 52, 'classifier__subsample': 0.8, 'classifier__n_estimators': 20,
         # 'classifier__min_samples_split': 82, 'classifier__min_samples_leaf': 27, 'classifier__max_features': None,
         #'classifier__max_depth': 100, 'classifier__learning_rate': 0.0001,
         #'classifier': GradientBoostingClassifier(learning_rate=0.0001, max_depth=100,
         #                                         min_samples_leaf=27, min_samples_split=82,
         #                                         n_estimators=20, random_state=42, subsample=0.8)}
        "pca__n_components": range(48,56,2),
        'classifier__n_estimators': range(2,50,4),
        'classifier__learning_rate': [0.0001],
        'classifier__max_depth': range(60, 140, 10),
        'classifier__min_samples_split': range(58, 102, 4),
        'classifier__min_samples_leaf': range(20, 40, 2),
        'classifier__max_features': ['auto', None],
        'classifier__subsample': [0.7,0.8,0.9],
        "classifier": [clf6]
    }

    param6b = {  # GRADIENT BOOSTING + kbest
        # reason I tried this classifier params:
        "kBest__k": range(2, 60, 10),
        'classifier__subsample': [0.6, 0.9],
        'classifier__n_estimators': range(5, 1000, 30),
        'classifier__min_samples_split': range(2, 100, 5),
        'classifier__min_samples_leaf': range(2, 100, 5),
        'classifier__max_features': ['auto', None, 'sqrt'],
        'classifier__max_depth': range(5, 1000, 30),
        'classifier__learning_rate': [0.01],
        "classifier": [clf6]
    }
    # GRADIENT BOOSTING ( NO PCA  NO KBEST)
    param6c = {
        # 'classifier__n_estimators': [5, 20, 35, 50, 65, 80, 95, 110, 125, 150, 200, 250, 350],
        'classifier__learning_rate': [0.001],
        'classifier__max_depth': range(5, 1000, 30),
        'classifier__min_samples_split': [1, 2, 9, 16, 24, 38],
        'classifier__min_samples_leaf': [1, 2, 9, 16, 24, 38],
        'classifier__max_features': ['auto', 'sqrt'],
        'classifier__subsample': [0.6, 0.95],
        "classifier": [clf6]
    }

    param7a = {  # CATBOOST CLASSIFIER + pca
        "pca__n_components": range(2, 60, 10),
        'classifier__n_estimators': range(2, 1000, 30),
        "classifier__learning_rate": [0.0001, 0.01, 0.1],
        'classifier__subsample': [0.1, 0.5, 0.7, 1.0],
        'classifier__max_depth': range(3, 500, 20),
        "classifier": [clf7]
    }
    param7b = {  # CATBOOST CLASSIFIER + kbest
        "kBest__k": range(4, 60, 8),
        'classifier__n_estimators': range(2, 1000, 30),
        "classifier__learning_rate": [0.0001, 0.01, 0.1],
        'classifier__subsample': [0.1, 0.5, 0.7, 1.0],
        'classifier__max_depth': range(3, 500, 20),
        "classifier": [clf7]
    }
    _4_layers = [(i, j, k, l) for i in range(3, 70, 3) for j in range(3, 70, 3) for k in range(3, 70, 3) for l in
                 range(3, 70, 3)]
    _5_layers = [(i, j, k, l, m) for i in range(30, 80, 5) for j in range(30, 80, 5) for k in range(30, 80, 5) for l in
                 range(30, 80, 5) for m in range(30, 80, 5)]
    _3_layers = [(i, j, k) for i in range(10, 70, 3) for j in range(10, 70, 3) for k in range(10, 70, 3)]
    _2_layers = [(i, j) for i in range(5, 100, 3) for j in range(5, 100, 3)]

    param8a = {  # MLPClassifier (neural network) + PCA
        #{'pca__n_components': 53, 'classifier__verbose': False, 'classifier__solver': 'sgd', 'classifier__max_iter': 1500, 'classifier__learning_rate': 'invscaling',
        # 'classifier__hidden_layer_sizes': (33, 44, 35, 39), 'classifier__alpha': 0.001, 'classifier__activation': 'relu', 'classifier': MLPClassifier(alpha=0.001, hidden_layer_sizes=(33, 44, 35, 39),
              # learning_rate='invscaling', max_iter=1500, random_state=42,
              # solver='sgd')}
        "pca__n_components": range(51,55),
        'classifier__hidden_layer_sizes': [(i, j, k, l) for i in range(29,35) for j in range(40,61,2) for k in range(25, 46, 2) for l in range(34, 45, 2)] + _5_layers,
        'classifier__activation':  ['relu'],
        'classifier__solver': ['sgd','adam'],
        'classifier__alpha': [0.001],
        # 'classifier__learning_rate': [ 'adaptive','constant','invscaling'],
        'classifier__learning_rate': ['invscaling','adaptive'],
        'classifier__max_iter': [500],
        'classifier__verbose': [False],  # details prints of loss
        "classifier": [clf8]
    }
    param8b = {  # MLPClassifier (neural network) + KBEST
        "kBest__k": range(4, 80, 8),
        'classifier__hidden_layer_sizes': [(i, j, k, l) for i in range(5, 50, 5) for j in range(5, 50, 5) for k in
                                           range(5, 50, 5)
                                           for l in range(5, 50, 5)],
        'classifier__activation': ['identity', 'logistic', 'tanh', 'relu'],
        'classifier__solver': ['adam', 'lbfgs', 'sgd'],
        'classifier__alpha': [0.0001, 0.01],
        'classifier__learning_rate': ['adaptive', 'constant', 'invscaling'],
        'classifier__max_iter': [3500],
        'classifier__verbose': [False],  # details prints of loss
        "classifier": [clf8]

    }

    param_3_classifier_only = {
        'classifier__gamma': range(20, 400, 10),
        'classifier__kernel': ['linear', 'rbf', 'sigmoid'],
        'classifier__C': range(20, 1000, 30),
        "classifier": [clf3]
    }
    param_6_classifier_only = {

        'classifier__n_estimators': range(2, 50, 4),
        'classifier__learning_rate': [0.0001],
        'classifier__max_depth': range(60, 140, 10),
        'classifier__min_samples_split': range(58, 102, 4),
        'classifier__min_samples_leaf': range(20, 40, 2),
        'classifier__max_features': ['auto', None],
        'classifier__subsample': [0.7, 0.8, 0.9],
        "classifier": [clf6]
    }
    param_8_classifier_only = {
        'classifier__hidden_layer_sizes': _2_layers + _3_layers + _4_layers + _5_layers,
        'classifier__activation': ['relu'],
        'classifier__solver': ['sgd', 'adam'],
        'classifier__alpha': [0.001],
        'classifier__learning_rate': ['adaptive','constant','invscaling'],
        'classifier__max_iter': [500],
        'classifier__verbose': [False],  # details prints of loss
        "classifier": [clf8]
    }
    sm = SMOTE(sampling_strategy='minority', random_state=42)
    # define the pipelines

    # pipe a - kbest
    # pipe b - pca
    pipe_smote_1a = imb_Pipeline(
        steps=[("smote", sm), ("scaler", scaler), ("pca", pca), ("classifier", param1a["classifier"][0])])
    pipe_smote_3a = imb_Pipeline(
        steps=[("smote", sm), ("scaler", scaler), ("pca", pca), ("classifier", param3a["classifier"][0])])
    pipe_smote_6a = imb_Pipeline(
        steps=[("smote", sm), ("scaler", scaler), ("pca", pca), ("classifier", param6a["classifier"][0])])
    pipe_smote_7a = imb_Pipeline(
        steps=[("smote", sm), ("scaler", scaler), ("pca", pca), ("classifier", param7a["classifier"][0])])
    pipe_smote_8a = imb_Pipeline(
        steps=[("smote", sm), ("scaler", scaler), ("pca", pca), ("classifier", param8a["classifier"][0])])
    pipe_smote_1b = imb_Pipeline(
        steps=[("smote", sm), ("scaler", scaler), ("kBest", kBest_selector), ("classifier", param1b["classifier"][0])])
    pipe_smote_3b = imb_Pipeline(
        steps=[("smote", sm), ("scaler", scaler), ("kBest", kBest_selector), ("classifier", param3b["classifier"][0])])
    pipe_smote_6b = imb_Pipeline(
        steps=[("smote", sm), ("scaler", scaler), ("kBest", kBest_selector), ("classifier", param6b["classifier"][0])])
    pipe_smote_7b = imb_Pipeline(
        steps=[("smote", sm), ("scaler", scaler), ("kBest", kBest_selector), ("classifier", param7b["classifier"][0])])
    pipe_smote_8b = imb_Pipeline(
        steps=[("smote", sm), ("scaler", scaler), ("kBest", kBest_selector), ("classifier", param8b["classifier"][0])])

    pipe1a = Pipeline(steps=[("scaler", scaler), ("pca", pca), ("classifier", param1a["classifier"][0])])
    pipe1b = Pipeline(steps=[("scaler", scaler), ("kBest", kBest_selector), ("classifier", param1b["classifier"][0])])
    pipe2a = Pipeline(steps=[("scaler", scaler), ("pca", pca), ("classifier", param2a["classifier"][0])])
    pipe2b = Pipeline(steps=[("scaler", scaler), ("kBest", kBest_selector), ("classifier", param2b["classifier"][0])])
    pipe3a = Pipeline(steps=[("scaler", scaler), ("pca", pca), ("classifier", param3a["classifier"][0])])
    pipe3b = Pipeline(steps=[("scaler", scaler), ("kBest", kBest_selector), ("classifier", param3b["classifier"][0])])
    pipe3_classifier_only = Pipeline(steps=[("scaler", scaler), ("classifier", param_3_classifier_only["classifier"][0])])
    pipe4a = Pipeline(steps=[("scaler", scaler), ("pca", pca), ("classifier", param4a["classifier"][0])])
    pipe4b = Pipeline(steps=[("scaler", scaler), ("kBest", kBest_selector), ("classifier", param4b["classifier"][0])])
    pipe5a = Pipeline(steps=[("scaler", scaler), ("pca", pca), ("classifier", param5a["classifier"][0])])
    pipe5b = Pipeline(steps=[("scaler", scaler), ("kBest", kBest_selector), ("classifier", param5b["classifier"][0])])
    # pipe5c = Pipeline(steps=[("scaler", scaler), ("classifier", param5c["classifier"][0])])
    pipe6a = Pipeline(steps=[("scaler", scaler), ("pca", pca), ("classifier", param6a["classifier"][0])])
    pipe6b = Pipeline(steps=[("scaler", scaler), ("kBest", kBest_selector), ("classifier", param6b["classifier"][0])])
    pipe6_classifier_only = Pipeline(steps=[("scaler", scaler), ("classifier", param_6_classifier_only["classifier"][0])])
    pipe7a = Pipeline(steps=[("scaler", scaler), ("pca", pca), ("classifier", param7a["classifier"][0])])
    pipe7b = Pipeline(steps=[("scaler", scaler), ("kBest", kBest_selector), ("classifier", param7b["classifier"][0])])
    # pipe7c = Pipeline(steps=[("scaler", scaler), ("classifier", param7c["classifier"][0])])
    pipe8a = Pipeline(steps=[("scaler", scaler), ("pca", pca), ("classifier", param8a["classifier"][0])])
    pipe8b = Pipeline(steps=[("scaler", scaler), ("kBest", kBest_selector), ("classifier", param8b["classifier"][0])])
    pipe8_classifier_only = Pipeline(steps=[("scaler", scaler), ("classifier", param_8_classifier_only["classifier"][0])])


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

# def balance_y_values(X, y, method):
#     """
#     making data set balanced in responders and non responders
#     """
#
#     if not args['classification']:
#         print("balancing y values supported only in calassification mode")
#         print("change args['classification'] to True and try over ")
#         exit()
#
#     print(f"balancing dataset (y values) using {method}...")
#     print(f"{X.shape[0]} y values\nvalue counts:\n{y.value_counts()}")
#
#     # select the method of balancing you want to use
#     if method == "undersample_majority":  # drop values from the larger category
#
#         data = X.join(y)
#         # print("balancing y values...")
#         # print(f"before balancing: {X.shape[0]} y values\nvalue counts:\n{y.value_counts()}")
#
#         # undersample responders
#         data = data.drop(data[data[y_name] == 1].sample(
#             frac=.3).index)  # LEAVE frac=.3 (matching args['both'] = True and args['both]= False)  drop a fraction of the responders to equalize num of responders and non responders
#         X = data.iloc[:, :-1]
#         y = data.iloc[:, -1]
#
#         return X, y
#
#     elif method == 'SMOTE':  # generate fake observations from the minority category
#
#         # Resampling the minority class. The strategy can be changed as required.
#         sm = SMOTE(sampling_strategy='minority', random_state=42)
#         # Fit the model to generate the data.
#         X, y = sm.fit_resample(X, y)
#         # oversampled = pd.concat([pd.DataFrame(oversampled_Y), pd.DataFrame(oversampled_X)], axis=1)
#
#     print(f"after balancing: {X.shape[0]} y values\nvalue counts:\n{y.value_counts()}")
#     return X, y


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

    print(f"**************************************************\n"
          f"Distribution of categorial variables in data:"
          f"**************************************************\n"
          "Gender:\n"
          f"X['gender'].value_counts(): {X['gender'].value_counts()}\n"
          f"X_train['gender'].value_counts():\n{X_train['gender'].value_counts()}\n"
          f"X_test['gender'].value_counts():\n{X_test['gender'].value_counts()}\n"
          f"\n**************************************************\n"
          "Treatment_group (coil):\n"
          f"X['Treatment_group'].value_counts(): {X['Treatment_group'].value_counts()}\n"
          f"X_train['Treatment_group'].value_counts():\n{X_train['Treatment_group'].value_counts()}\n"
          f"X_test['Treatment_group'].value_counts():\n{X_test['Treatment_group'].value_counts()}\n"
          "Response to treatment:\n"
          f"\n**************************************************\n"
          f"['6-weeks_HDRS21_class'].value_counts():\n{y['6-weeks_HDRS21_class'].value_counts()}\n"
          f"y_train['6-weeks_HDRS21_class'].value_counts():\n{y_train['6-weeks_HDRS21_class'].value_counts()}\n"
          f"y_test['6-weeks_HDRS21_class'].value_counts():\n{y_test['6-weeks_HDRS21_class'].value_counts()}\n")

    # # fix imbalance in train set
    # if args['balance_y_values']:
    #     X_train, y_train = balance_y_values(X_train, y_train, method='SMOTE')
    # *******************************
    if args['classification']:
        if args['lite_mode']:  # just for debugging. using one small grid
            # param_pipe_list = [[param3a, pipe_smote_3a]] # CHECKED
            # param_pipe_list = [[param6a, pipe_smote_6a]] # CHECKED
            # param_pipe_list = [[param7a, pipe_smote_7a]] # CATBOOST - BUGS
            # param_pipe_list = [[param8a, pipe_smote_8a]] # CHECKED

            # param_pipe_list = [[param3b, pipe_smote_3b]] # CHECKED
            # param_pipe_list = [[param6b, pipe_smote_6b]] # CHECKED
            # param_pipe_list = [[param7b, pipe_smote_7b]] # CATBOOST - BUGS

            # param_pipe_list = [[param8b, pipe_smote_8b]] # CHECKED

            # param_pipe_list = [[param_3_classifier_only, pipe3_classifier_only]]
            # param_pipe_list = [[param_6_classifier_only, pipe6_classifier_only]]
            param_pipe_list = [[param_8_classifier_only, pipe8_classifier_only]]
        # ********************************
        else:  # more than one model
            # pipe is represent the steps we want to execute, param represents which args we want to execute with
            param_pipe_list = []  # put all the pipe and param paris you want

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

        # # cross validation search

        if args['exhaustive_grid_search']:  # exhausitve search
            print("~~~~~~~~~~ EXHAUSTIVE SEARCH CV ~~~~~~~~~~~")
            search = GridSearchCV(param_grid=param, estimator=pipe, cv=args["cv"], verbose=3, scoring=scorer(),
                                  refit=True)
            choice_options = [len(val) for val in param.values()]
            args['n_iter'] = reduce(lambda x, y: x * y, choice_options)  # multiply all choice options
            search.fit(X_train.astype(float), y_train)
        else:  # randomized search
            print("~~~~~~~~~~ RANDOMIZED SEARCH CV ~~~~~~~~~~")
            search = RandomizedSearchCV(estimator=pipe, param_distributions=param, n_iter=args["n_iter"], cv=args["cv"],
                                        n_jobs=args['n_jobs'], verbose=3, random_state=args['rs'], scoring=scorer(),
                                        refit=True)

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
        print_conclusions(X_train, pipe, search, best_cv_iter_yts_list_ndarray, best_cv_iter_yps_list_ndarray,folder)

print(f"<<<<<<<<<<<<<<<<<<<<< GSCVrunner.py finished successfuly<<<<<<<<<<<<<<<<<<<<<")
if args['stdout_to_file']:
    log_file.close()
plt.show()
