
import math
import shutil
import subprocess
from functools import reduce

import sklearn.metrics
from sklearn.metrics import roc_curve
from matplotlib import pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.svm import SVC
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import classification_report
import datetime
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import metrics
from sklearn.feature_selection import SelectKBest, f_classif, f_regression, mutual_info_classif
from sklearn.pipeline import Pipeline
from sklearn.metrics import fbeta_score
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from catboost import CatBoostClassifier
import warnings
import main_caller_r2
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, roc_auc_score, precision_score, recall_score
from sklearn.metrics import make_scorer
import random
import os
from runArguments import args
from sklearn.experimental import enable_halving_search_cv  # noqa
from imblearn.over_sampling import SMOTE
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


def get_pcs_num_explains_p_ratio_of_var(p_ratio, ratios):
    assert (0 < p_ratio < 1)
    ratio_sums = [0 for r in ratios]
    ratio_sums[0] = ratios[0]
    pc_index_p_ratio_of_var_explained = -1 if ratios[
                                                  0] < p_ratio else 0  # first pc which togeteher with previous pcs explains over p_ratio of var
    # print pcs var explaining ability
    for i in range(1, len(ratios)):
        ratio_sums[i] = ratio_sums[i - 1] + ratios[i]
        if ratio_sums[i] > p_ratio:
            pc_index_p_ratio_of_var_explained = i
            break
    return pc_index_p_ratio_of_var_explained


def print_conclusions(df=None, pipe=None, search=None,
                      folder="out_folder", mode=None, y_true=None, y_pred=None):
    """
    Prints a detailed report for some pipeline results (train or test)
    """
    label = mode # cv / train / test
    if mode == 'cv':
        print(f"-----------------------\n New {label} report \n-----------------------")
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
        print(
            f"* CV mean_test_score {args['scoring_method']} ( over {args['cv']} folds - (cv's best score for best hyperparametes): %.3f +/- %.3f (see score func in hyperparams) " % (
                score_mean, score_std), "\n")


    labels = ['Non-responsive', 'Responsive']
    cm = confusion_matrix(y_true, y_pred)
    # Create a figure and plot the confusion matrix
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)

    # Set labels and title
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=labels, yticklabels=labels,
           xlabel='Predicted label', ylabel='True label',
           title='Confusion Matrix')

    # # Rotate the x-axis labels if needed
    # plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Loop over data dimensions and create text annotations
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

    # Save the figure to a PNG file
    figpath = os.path.join(folder, f'{label}_cm.png')
    # fig.set_size_inches(8, 12)  # Adjust the height (6 inches) as needed
    fig.savefig(figpath)  # Replace 'path/to/save' with the desired file path

    cm_with_legend = str(cm) + "\n" + "[[TN FP\n[FN TP]]"
    print("* Confusion matrix: \n", cm_with_legend)

    # calculate Response rate for X_train:
    if mode == 'cv':
        y_train_responders_cnt = 0
        positive_label = 1  # can vary in different experiments
        for y_val in y_train.values:
            if y_val == positive_label:
                y_train_responders_cnt += 1
        print("* Response rate: ", y_train_responders_cnt / len(y_train))

    true_count = cm[0][0] + cm[1][1]
    false_count = cm[0][1] + cm[1][0]
    total = true_count + false_count
    precision = cm[1][1] / (cm[1][1] + cm[0][1])
    recall = cm[1][1] / (cm[1][1] + cm[1][0])
    accuracy = true_count / total
    f1 = 2 * (precision * recall) / (precision + recall)
    print("* Precision: ", precision)
    print("* Recall: ", recall)
    print("* Accuracy: ", accuracy)
    print("* F1: ", f1)
    print(f"* F-Beta (beta = {args['beta']}): ", fbeta_score(y_true, y_pred, beta=args['beta']))  # beta = weight given to recall (weight given to precision is always 1)

    # save cross val scores and params in tuning.csv for tracking
    # csv column names and values in a row:
    if mode == 'cv':
        d = {
            "date": str(datetime.date.today()),  # date, hour
            "classifier": str(pipe.named_steps['classifier']),
            "pipe_named_steps": str(pipe.named_steps),  # all pipe steps, including the classifier as estimator
            "best_params": str(search.best_params_),
            #  "best_index": search.best_index_ ,
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
        file_path = os.path.join(folder, 'tuning.csv')
        open(file_path, 'a').close()
        tmp.to_csv(file_path, index=False)
        print("train CV report saved to  ", file_path)

    save_roc_auc = True
    if mode in ("test", "train"):
        if save_roc_auc:
            # roc auc
            if mode == "test":
                y_pred_prob_positives = search.predict_proba(X_test)[:, 1]
                # label = "Test"
            elif mode == "train":
                y_pred_prob_positives = search.predict_proba(X_train)[:, 1]
                # label = "Train"

            fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred_prob_positives)

            # Calculate the Area Under the ROC Curve (AUC)
            roc_auc = metrics.auc(fpr, tpr)

            # Create a figure and plot the ROC curve
            plt.figure()
            plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (AUC = %0.2f)' % roc_auc)
            plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
            plt.xlim([-0.02, 1.02])
            plt.ylim([-0.02, 1.02])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic (ROC) Curve')
            plt.legend(loc='lower right')

            # Save the figure to a PNG file
            figpath = os.path.join(folder, f'roc_auc_{label}.png')
            plt.savefig(figpath)

            # Show the plot (optional)
            plt.show()
    print(f"-----------------------\n End of {label} report \n-----------------------", '\n' * 3)


def ci(X_test, y_test,model,score):
    # dummy = sklearn.dummy.DummyClassifier(strategy="constant", constant=1)
    dummy_predictions =  [1]*len(y_test)
    scores_model = []
    scores_dummy = []

    for n in range(500):
        random_indices = np.random.choice(X_test.index, size=len(X_test), replace=True)
        X_test_new = X_test.loc[random_indices,:]
        y_test_new = y_test.loc[random_indices,:]
        if score == 'precision':
            scores_model.append(precision_score(y_test_new, model.predict(X_test_new)))
            scores_dummy.append(precision_score(y_test_new,dummy_predictions))
        elif score == 'accuracy':
            scores_model.append(accuracy_score(y_test_new, model.predict(X_test_new)))
            scores_dummy.append(accuracy_score(y_test_new, dummy_predictions))

        elif score == 'f1':
            scores_model.append(f1_score(y_true=y_test_new, y_pred = model.predict(X_test_new)))
            scores_dummy.append(f1_score(y_true=y_test_new,  y_pred = dummy_predictions))

    return np.quantile(scores_model,[0.025,0.975]),np.quantile(scores_dummy,[0.025,0.975])


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
        if args['beta']:
            beta = args['beta']
        else:
            beta = 0.5
        cvscore = fbeta_score(y_true, y_pred,
                            beta)  # beta = weight given to recall (weight given to precision is always 1)

    if cvscore is None:
        print("problem - cvscore is nan.check y_true, y_pred...")
        exit()

    # y_true = y_true[y_name].values  # change from df to ndarray for printing nicely
    param_choice_idx = total_folds[0] // len(choice_scores)

    idx = total_folds[0] % len(choice_scores)  # num of folds elapsed % num of folds in each param choice
    choice_scores[idx] = cvscore
    max_param_choice_idx = args['n_iter'] - 1
    max_split_index = total_splits - 1

    if idx == 0:
        param_choice_str = f"Parameter choice num {param_choice_idx} / {max_param_choice_idx} - starting..."
        print(param_choice_str)

    print(f"{total_folds[0]} / {max_split_index} splits counted in cross val search ")
    print("fold's true y \n", y_true)
    print("fold's predicted y\n", y_pred)
    print(f"scoring metric: {my_scorer}, score: {cvscore} ")
    print(">>>")
    predicted_correctly = len(y_true[y_true == y_pred])
    predicted_in_total = len(y_true)
    print(f"predicted correctly / predicted_in_total = {predicted_correctly} / {predicted_in_total}")
    print("<<<")
    if idx == len(choice_scores) - 1:  # calculated score for all folds in current parmeter coice
        choice_avg_score = np.mean(choice_scores)
        choice_avg_score_str = f"In parameter choice num {param_choice_idx} / {max_param_choice_idx} avg score was: {choice_avg_score}."

        improvement_report_str = None
        if choice_avg_score > best_score_by_now[0]:
            best_score_by_now[0] = choice_avg_score
            improvement_report_str = f"New best score is {best_score_by_now[0]}"
            # parmam_str = f"chosen parameters: {pipe.get_params()}" # param configuration chose

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


def get_top_k_contributions_and_contributors(k=3, all_pcs=None):
    # 1.Find k features with max contribution

    feature_contributions = [0 for _ in
                             X_train.columns]  # contribution to (influence on) the explained var from each features
    for i, pc in enumerate(all_pcs):
        for j, contribution in enumerate(pc):
            feature_contributions[j] += abs(contribution)
    # Sort the array while preserving the original indices
    sorted_arr = sorted(enumerate(feature_contributions), key=lambda x: x[1])[
                 ::-1]  # sorted tuples of item and original index
    sorted_contributions = [sorted_arr[i][1] for i in range(len(sorted_arr))]
    sum_sorted_contributions = sum(sorted_contributions)
    for i in range(len(sorted_contributions)):
        sorted_contributions[i] = sorted_contributions[i] / sum_sorted_contributions
    sum_sorted_contributions_scaled = sum(sorted_contributions)
    # Print the sorted array with original indices
    top_k_contributions = sorted_contributions[:k]
    top_k_contributors = [""] * k
    for i in range(k):
        contributor_col_index = sorted_arr[i][0]
        contributor = X_train.columns[contributor_col_index]
        top_k_contributors[i] = contributor
        print(f"Feature (contributor) i = {i}: name =  {contributor}, Value (contribution) = {top_k_contributions[i]}")
    print(f"top_k_contribution sum : {sum(top_k_contributions)}")
    print(
        f"=> {k} most contributing features (out of {len(sorted_contributions)} features in total) are explaining  {(sum(top_k_contributions) / sum_sorted_contributions_scaled):.5f} of variance")
    return top_k_contributions, top_k_contributors


def make_fig_contributions(top_contributors, top_contributions, k):
    k = len(top_contributions)

    fig = plt.figure(figsize=(14, 12))
    labels = top_contributors + ['expecteds contribution']
    # Generate gradual colors for the bars
    colors = plt.cm.viridis(np.linspace(0, 1, len(top_contributions) + 1))
    num_features_data = 1388
    expected_contrib = [1 / num_features_data]
    # Plot the bar graph
    contributions = top_contributions + expected_contrib
    plt.bar(range(len(contributions)), contributions, color=colors)

    plt.title('Contribution')
    # Set the x-axis labels
    plt.xticks(range(len(labels)), labels, rotation=45, ha='right')

    figpath = os.path.join(folder, 'contributions_pca_x_train.png')
    fig.savefig(figpath)  # Replace 'path/to/save' with the desired file path


def make_fig_heatmap(X):
    # Calculate correlation matrix
    corr_matrix = X.corr()

    # Create heatmap using seaborn
    fig = plt.figure(figsize=(17, 17))  # Adjust figure size as needed
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', square=True, xticklabels=True, yticklabels=True,
                annot_kws={"fontsize": 9})
    plt.title('Correlation Heatmap')
    figpath = os.path.join(folder, 'corellations_contributors_pcs_x_train.png')
    fig.savefig(figpath)
    # Display the heatmap


def make_fig_kbest(features, feature_scores):
    fig = plt.figure(figsize=(14, 12))
    labels = features
    # Generate gradual colors for the bars
    colors = plt.cm.viridis(np.linspace(0, 1, len(features)))

    # Plot the bar graph
    plt.bar(range(len(labels)), feature_scores, color=colors)

    plt.title(f'p values of best k({k}) features (train set)')

    # Set the x-axis labels
    plt.xticks(range(len(labels)), labels, rotation=45, ha='right')
    figpath = os.path.join(folder, 'kbest.png')
    fig.savefig(figpath)


# ******************************************* MAIN ****************************************************

print(" >>>>>>>>>>>>>>>>>>>>> STARTING MAIN OF GSCVrunner.py >>>>>>>>>>>>>>>>>>>>>")
now = datetime.datetime.now()
folder = now.strftime("%Y-%m-%d %H_%M_%S") + args['output_folder_label']
if not os.path.isdir(folder):
    os.mkdir(folder)
shutil.copyfile(os.path.join(os.getcwd(),"GSCVrunner_new.py"),os.path.join(folder,"GSCVrunner_new.py"))
shutil.copyfile(os.path.join(os.getcwd(),"runArguments.py"),os.path.join(folder,"runArguments.py"))

if args['stdout_to_file']:
    logfile_name = os.path.join(folder, 'stdout.txt')
    print(f"see stdout in {logfile_name}")
    if os.path.exists(logfile_name):
        os.remove(logfile_name)
    log_file = open(logfile_name, 'a')
    sys.stdout = sys.stderr = log_file
search_statistics = os.path.join(folder, "search_statistics.txt")
if os.path.exists(search_statistics):
    os.remove(search_statistics)
if args['classification']:
    y_name = '6-weeks_HDRS21_class'  # classification problem
else:
    y_name = "6-weeks_HDRS21_change_rate"  # regression problem

X, y = main_caller_r2.get_X_y(y_name, args["X_version"])  # X and y initialization
pca = PCA()
scaler = StandardScaler()
kBest_selector = SelectKBest()
sm = SMOTE(sampling_strategy='minority', random_state=42)

if not os.path.exists(os.path.join(folder, 'X.csv')):
    X.to_csv(os.path.join(folder, 'X.csv'))
if not os.path.exists(os.path.join(folder, 'y.csv')):
    y.to_csv(os.path.join(folder, 'y.csv'))

X.reset_index(inplace=True, drop=True)

if args['classification']:
    y[y_name] = y[y_name].replace({"responsive": 1, "non_responsive": 0})

if args["age_under_50"]:
    df = X.join(y)
    df = df[df['age'] < 50]
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1:]

if args['classification']:
    clf1 = LogisticRegression(random_state=args["rs"])
    clf2 = KNeighborsClassifier()
    clf3 = SVC(probability=True, random_state=args["rs"])
    clf4 = DecisionTreeClassifier(random_state=args["rs"])
    clf5 = RandomForestClassifier(random_state=args["rs"])
    clf6 = GradientBoostingClassifier(random_state=args["rs"])
    clf7 = CatBoostClassifier(random_state=args["rs"], logging_level='Silent')
    clf8 = MLPClassifier(random_state=args["rs"])

    # LOGISTIC REGRESSION + pca
    param1a = {
        "pca__n_components": [1, 3, 5, 7, 9, 11],
        "classifier__C": [0.001, 0.01, 0.1, 1, 10, 100],
        "classifier__penalty": ['l2'],
        "classifier": [clf1]
    }
    # LOGISTIC REGRESSION + kbest
    param1b = {
        "classifier__C": [0.45, 0.47, 0.5, 0.52, 0.55, 0.6],  # classifier (logistic regression) param 'C' for tuning
        "kBest__k": range(15, 17),  # selctKbest param 'k'for tuning. must be  <= num of features
        'classifier__penalty': ['l2'],
        "kBest__score_func": [mutual_info_classif],  # selctKbest param 'score_func'for tuning
        "classifier": [clf1]
    }
    # knn + pca
    param2a = {
        "pca__n_components": [53],
        "classifier__n_neighbors": range(5, 150, 3),
        "classifier__weights": ['uniform', 'distance'],
        "classifier__algorithm": ['auto', 'ball_tree', 'kd_tree', 'brute'],
        # "classifier__p": [2],
        "classifier": [clf2]
    }
    # KNN + kbest
    param2b = {
        "classifier__n_neighbors": range(1, 90, 3),
        "classifier__weights": ['uniform', 'distance'],
        "classifier__algorithm": ['auto', 'ball_tree', 'kd_tree', 'brute'],
        "classifier__leaf_size": range(3, 80, 3),
        "classifier__p": [2],
        "kBest__k": range(4, 100, 3),
        "kBest__score_func": [f_classif, mutual_info_classif],  # selctKbest param 'score_func'for tuning
        "classifier": [clf2]
    }
    # SVC + pca
    param3a = {
        "pca__n_components": range(2, 55, 10),
        'classifier__gamma': range(20, 400, 10),
        'classifier__kernel': ['linear', 'rbf', 'sigmoid'],
        'classifier__C': range(20, 1000, 30),
        "classifier": [clf3]
    }
    # SVC + kbest
    param3b = {
        "kBest__k": range(2, 55, 10),  # k should be smaller than num of features in X
        'classifier__gamma': range(5, 1000, 30),
        'classifier__kernel': ['linear', 'rbf', 'sigmoid'],
        'classifier__C': range(5, 1500, 20),
        "classifier": [clf3]
    }
    # DECISION TREE + pca
    param4a = {
        "pca__n_components": range(2, 55, 10),
        'classifier__max_leaf_nodes': range(5, 1000, 30),
        'classifier__max_depth': range(5, 1000, 30),
        'classifier__criterion': ['entropy', 'gini'],
        'classifier__min_samples_split': range(2, 100, 5),
        'classifier__min_samples_leaf': range(20, 100, 5),
        "classifier": [clf4]
    }
    # DECISION TREE + kbest
    param4b = {
        "kBest__k": range(470, 800, 10),
        'classifier__max_leaf_nodes': range(78, 90, 2),
        'classifier__max_depth': range(260, 280, 3),
        'classifier__criterion': ['gini'],
        'classifier__min_samples_split': [2, 3],
        'classifier__min_samples_leaf': [1, 2],
        "classifier": [clf4]

    }
    # RANDOM FOREST NO DIMENSION REDUCTION
    param5 = {
        "classifier__n_jobs": [-1],
        #
        # "classifier__bootstrap": [True, False],
        # 'classifier__n_estimators': [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000],  # number of trees
        # "classifier__max_depth": [None, 10, 20, 50, 100, 200],
        # "classifier__max_features": ['auto', 'sqrt'],
        # 'classifier__min_samples_split': [2, 5, 10],
        # 'classifier__min_samples_leaf': [1, 2, 4, 6, 10],
        "classifier": [clf5]
    }

    # RANDOM FOREST + pca
    param5a = {
        "pca__n_components": [71],
        "classifier__n_jobs": [-1],
#        'classifier__n_estimators': [5000],  # number of trees
        "classifier__max_depth": [75],
        "classifier__max_features": [2,10,100,'auto'],
        'classifier__min_samples_split': [7],
        'classifier__min_samples_leaf': [5],
        "classifier": [clf5]

    #     {'pca__n_components': 71, 'classifier__min_samples_split': 7, 'classifier__min_samples_leaf': 5,
    #      'classifier__max_features': 2, 'classifier__max_depth': 75, 'classifier__bootstrap': True,
    #      'classifier': RandomForestClassifier(max_depth=75, max_features=2, min_samples_leaf=5,
    #                                           min_samples_split=7, random_state=42)}
    #
    }
    # RANDOM FOREST + kbest
    param5b = {
        "kBest__k": [30],
        'classifier__n_estimators': [500],  # number of trees
        "classifier__n_jobs": [-1],
        "classifier__max_depth": [None, 3, 6, 10, 50, 100],
        "classifier__max_features": ['sqrt'],
        'classifier__min_samples_split': [2, 5, 10,40],
        'classifier__min_samples_leaf': [2,3,4,7,10],
        "classifier": [clf5]
    }
    # GRADIENT BOOSTING + pca
    param6a = {

        'pca__n_components': [54], 'classifier__subsample': [0.8], 'classifier__n_estimators': [20],
         'classifier__min_samples_split': [82],
        'classifier__min_samples_leaf': [27],
         'classifier__max_features': [None], 'classifier__max_depth': [100], 'classifier__learning_rate': [0.0001],
         'classifier': [clf6]
    }

    # GRADIENT BOOSTING + kbest
    param6b = {
        # reason I tried this classifier params:
        "kBest__k": [15],
        'classifier__subsample': [0.9],
        'classifier__n_estimators': range(10, 30, 5),
        'classifier__min_samples_split': range(60, 100, 5),
        'classifier__min_samples_leaf': range(10, 40, 5),
        'classifier__max_features': [None],
        'classifier__max_depth': range(80, 120, 5),
        'classifier__learning_rate': [0.0001],
        "classifier": [clf6]
    }
    param7 = {
        "classifier": [clf7]
    }
    # CATBOOST CLASSIFIER + pca
    param7a = { # https://docs.aws.amazon.com/sagemaker/latest/dg/catboost-tuning.html
        "pca__n_components": [53],
        "classifier": [clf7],
        "classifier__depth": [4,6,8,10],
        "classifier__l2_leaf_reg": [2,4,6,8, 10],
        "classifier__learning_rate": [0.001, 0.005, 0.01],
        "classifier__random_strength": [0.001, 0.005, 0.01]
    }

    # CATBOOST CLASSIFIER + kbest
    param7b = {
        "kBest__k": [30,40,50],
        "classifier": [clf7],
        "classifier__depth": [4,6,8,10],
        "classifier__l2_leaf_reg": [2,4,6,8, 10],
        "classifier__learning_rate": [0.001, 0.005, 0.01],
        "classifier__random_strength": [0.001, 0.005, 0.01]
    }
    # MLP CLASSIFIER LAYERS FOR SEARCH
    _4_layers = [(i, j, k, l) for i in range(3, 70, 3) for j in range(3, 70, 3) for k in range(3, 70, 3) for l in
                 range(3, 70, 3)]
    _5_layers = [(i, j, k, l, m) for i in range(30, 80, 5) for j in range(30, 80, 5) for k in range(30, 80, 5) for l in
                 range(30, 80, 5) for m in range(30, 80, 5)]
    _3_layers = [(i, j, k) for i in range(10, 70, 3) for j in range(10, 70, 3) for k in range(10, 70, 3)]
    _2_layers = [(i, j) for i in range(1, 200, 4) for j in range(1, 200, 4)]
    # MLPClassifier + pca
    param8 = {
        "classifier": [clf8]
    }

    param8a = {

        "pca__n_components": [53],
        'classifier__hidden_layer_sizes':[(70, 36, 25, 40, 73)],
        # 'classifier__activation': ['relu'],
        'classifier__solver': ['sgd'],
        'classifier__alpha': [0.001],
        'classifier__learning_rate': ['invscaling'],
        'classifier__max_iter': [1200],
        'classifier__verbose': [False],  # details prints of loss
        "classifier": [clf8]

    }
    # MLPClassifier + KBEST
    param8b = {
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
    # new grids:
    param2a_smote_no_dimension_red = {
        "classifier__n_neighbors": range(5, 150, 3),
        "classifier__weights": ['uniform', 'distance'],
        "classifier__algorithm": ['auto', 'ball_tree', 'kd_tree', 'brute'],
        "classifier": [clf2]
    }
    param_2_classifier_only = {
        "classifier__n_neighbors": range(30, 50, 2),
        "classifier__weights": ['uniform', 'distance'],
        "classifier__algorithm": ['auto', 'ball_tree', 'kd_tree', 'brute'],
        "classifier": [clf2]

    }
    param_3_classifier_only = {
        'classifier__gamma': range(20, 400, 10),
        'classifier__kernel': ['linear', 'rbf', 'sigmoid'],
        'classifier__C': range(20, 1000, 30),
        "classifier": [clf3]
    }
    param_6_classifier_only = {

        "classifier": [clf6]
    }
    param_8_classifier_only = {
        'classifier__hidden_layer_sizes': _2_layers,
        'classifier__activation': ['relu'],
        'classifier__solver': ['sgd', 'adam'],
        'classifier__alpha': [0.01, 0.001, 0.0001],
        'classifier__learning_rate': ['adaptive', 'invscaling'],
        'classifier__max_iter': [700],
        'classifier__verbose': [False],  # details prints of loss
        "classifier": [clf8]
    }
    param_8_dimension_red_only = {
        # 'classifier__hidden_layer_sizes': _3_layers,
        # {'pca__n_components': 35, 'classifier__verbose': False, 'classifier__solver': 'sgd',
        #  'classifier__max_iter': 700, 'classifier__learning_rate': 'invscaling',
        #  'classifier__hidden_layer_sizes': (46, 19, 19), 'classifier__alpha': 0.001, 'classifier__activation': 'relu',
        #  'classifier': MLPClassifier(alpha=0.001, hidden_layer_sizes=(46, 19, 19),
        #                              learning_rate='invscaling', max_iter=700, random_state=42,
        #                              solver='sgd')}

        # * Best Hyperparametes picked in cross validation: (cv's best score):
        # {'pca__n_components': 10, 'classifier__verbose': False, 'classifier__solver': 'sgd', 'classifier__max_iter': 800,
        # 'classifier__learning_rate': 'invscaling', 'classifier__hidden_layer_sizes': (33, 10, 10),
        # 'classifier__alpha': 0.01, 'classifier__activation': 'relu', 'classifier': MLPClassifier(alpha=0.01, hidden_layer_sizes=(33, 10, 10),
        #   learning_rate='invscaling', max_iter=800, random_state=42,
        #   solver='sgd')}

        # {'pca__n_components': 10, 'classifier__verbose': False, 'classifier__solver': 'sgd',
        # 'classifier__max_iter': 800, 'classifier__learning_rate': 'invscaling', 'classifier__hidden_layer_sizes': (33, 10, 10), 'classifier__alpha': 0.01, 'classifier__activation': 'relu', 'classifier': MLPClassifier(alpha=0.01, hidden_layer_sizes=(33, 10, 10),
        # learning_rate='invscaling', max_iter=800, random_state=42,
        # solver='sgd')}

        "pca__n_components": [100],
        'classifier__hidden_layer'
        '_sizes': [(1, 1, 1)] + _3_layers,
        'classifier__activation': ['relu'],
        'classifier__solver': ['sgd'],
        'classifier__alpha': [0.005],
        'classifier__learning_rate': ['invscaling'],
        'classifier__max_iter': [800],
        'classifier__verbose': [False],  # details prints of loss
        "classifier": [clf8]
    }
    param6_smote_no_dimension_red = {
        'classifier__n_estimators': range(10, 50, 4),
        'classifier__learning_rate': [0.001],
        'classifier__max_depth': range(60, 140, 10),
        'classifier__min_samples_split': range(2, 102, 4),
        'classifier__min_samples_leaf': range(2, 40, 2),
        'classifier__max_features': ['auto', None],
        'classifier__subsample': [0.7, 0.8, 0.9],
        "classifier": [clf6]
    }

    # Pipelines
    pipe1a = Pipeline(steps=[("scaler", scaler), ("pca", pca), ("classifier", param1a["classifier"][0])])
    pipe1b = Pipeline(steps=[("scaler", scaler), ("kBest", kBest_selector), ("classifier", param1b["classifier"][0])])
    pipe2a = Pipeline(steps=[("scaler", scaler), ("pca", pca), ("classifier", param2a["classifier"][0])])
    pipe2b = Pipeline(steps=[("scaler", scaler), ("kBest", kBest_selector), ("classifier", param2b["classifier"][0])])
    pipe2_classifier_only = Pipeline(
        steps=[("scaler", scaler), ("classifier", param_2_classifier_only["classifier"][0])])
    pipe3a = Pipeline(steps=[("scaler", scaler), ("pca", pca), ("classifier", param3a["classifier"][0])])
    pipe3b = Pipeline(steps=[("scaler", scaler), ("kBest", kBest_selector), ("classifier", param3b["classifier"][0])])
    pipe3_classifier_only = Pipeline(
        steps=[("scaler", scaler), ("classifier", param_3_classifier_only["classifier"][0])])
    pipe4a = Pipeline(steps=[("scaler", scaler), ("pca", pca), ("classifier", param4a["classifier"][0])])
    pipe4b = Pipeline(steps=[("scaler", scaler), ("kBest", kBest_selector), ("classifier", param4b["classifier"][0])])
    pipe5a = Pipeline(steps=[("scaler", scaler), ("pca", pca), ("classifier", param5a["classifier"][0])])
    pipe5b = Pipeline(steps=[("scaler", scaler), ("kBest", kBest_selector), ("classifier", param5b["classifier"][0])])
    pipe5 = Pipeline(steps=[("scaler", scaler), ("classifier", param5["classifier"][0])])
    pipe6a = Pipeline(steps=[("scaler", scaler), ("pca", pca), ("classifier", param6a["classifier"][0])])
    pipe6b = Pipeline(steps=[("scaler", scaler), ("kBest", kBest_selector), ("classifier", param6b["classifier"][0])])
    pipe6_classifier_only = Pipeline(
        steps=[("scaler", scaler), ("classifier", param_6_classifier_only["classifier"][0])])
    pipe7a = Pipeline(steps=[("scaler", scaler), ("pca", pca), ("classifier", param7a["classifier"][0])])
    pipe7b = Pipeline(steps=[("scaler", scaler), ("kBest", kBest_selector), ("classifier", param7b["classifier"][0])])
    pipe7 = Pipeline(steps=[("scaler", scaler), ("classifier", param7["classifier"][0])])
    pipe8a = Pipeline(steps=[("scaler", scaler), ("pca", pca), ("classifier", param8a["classifier"][0])])
    pipe8b = Pipeline(steps=[("scaler", scaler), ("kBest", kBest_selector), ("classifier", param8b["classifier"][0])])
    pipe8_classifier_only = Pipeline(
        steps=[("scaler", scaler), ("classifier", param_8_classifier_only["classifier"][0])])
    pipe8_dimension_red_only = Pipeline(
        steps=[("scaler", scaler), ("pca", pca), ("classifier", param_8_dimension_red_only["classifier"][0])])
    pipe_smote_1a = imb_Pipeline(
        steps=[("smote", sm), ("scaler", scaler), ("pca", pca), ("classifier", param1a["classifier"][0])])
    pipe_smote_2a = imb_Pipeline(
        steps=[("smote", sm), ("scaler", scaler), ("pca", pca), ("classifier", param3a["classifier"][0])])
    pipe_smote_2a_no_dimension_reduction = imb_Pipeline(
        steps=[("smote", sm), ("scaler", scaler), ("classifier", param2a_smote_no_dimension_red["classifier"][0])])
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
    pipe_smote_5a = imb_Pipeline(
        steps=[("smote", sm), ("scaler", scaler), ("pca", pca), ("classifier", param5a["classifier"][0])])

    pipe_5_scaler_smote_rf = imb_Pipeline(
        steps=[("scaler", scaler), ("smote", sm), ("classifier", param5["classifier"][0])])
    pipe_smote_5b = imb_Pipeline(
        steps=[("scaler", scaler), ("smote", sm), ("kBest", kBest_selector), ("classifier", param5a["classifier"][0])])

    pipe_smote_6b = imb_Pipeline(
        steps=[("smote", sm), ("scaler", scaler), ("kBest", kBest_selector), ("classifier", param6b["classifier"][0])])
    pipe_smote_7b = imb_Pipeline(
        steps=[("smote", sm), ("scaler", scaler), ("kBest", kBest_selector), ("classifier", param7b["classifier"][0])])
    pipe_smote_8b = imb_Pipeline(
        steps=[("smote", sm), ("scaler", scaler), ("kBest", kBest_selector), ("classifier", param8b["classifier"][0])])
    pipe_smote_6_smote_no_dimension_red = imb_Pipeline(
        steps=[("smote", sm), ("scaler", scaler), ("classifier", param6_smote_no_dimension_red["classifier"][0])])
# Grid Search
splits = []  # each split is a list of : [X_train, X_test,y_train, y_test]
if (args["split_rows"] == 'normal'):  # regular test train split  =  don't drop subjects:
    if args['classification']:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args['test_size'], random_state=args["rs"],
                                                            shuffle=True, stratify=y)
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args['test_size'], random_state=args["rs"])
    splits.append([X_train, X_test, y_train, y_test])

elif args["split_rows"] in ['h1', 'h7', 'h1h7']:  # split by 'Treatment_group' (device - h1/h7)
    print("splitting by h1 h7 : ")
    if args['split_rows'] in ['h1', 'h1h7']:  # use h1
        # h1 subjects only
        df1 = X.join(y)
        df1 = df1[df1['Treatment_group'] == 0]
        print("new data- only the rows where column 'Treatment_group is 0:")
        X_tmp = df1.iloc[:, :-1]
        y_tmp = df1.iloc[:, -1]
        if args['classification']:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args['test_size'],
                                                                random_state=args["rs"], shuffle=True, stratify=y_tmp)
        else:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args['test_size'],
                                                                random_state=args["rs"], shuffle=True)
        splits.append([X_train, X_test, y_train, y_test])
    if args['split_rows'] in ['h7', 'h1h7']:  # use h7
        # h7 subjects only
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
        splits.append([X_train, X_test, y_train, y_test])

for config in splits:
    for i in range(4):
        if isinstance(config[i], pd.Series):  # convert to data frame to make scorer work
            config[i] = config[i].to_frame()
    X_train, X_test, y_train, y_test = config[0], config[1], config[2], config[3]

    add_train_data_reports = False
    if add_train_data_reports:
        # Analyze X_train with PCA:
        report_pca = PCA()
        X_train_for_pca = scaler.fit_transform(X_train.copy())
        report_pca.fit(X_train_for_pca)
        all_pcs = report_pca.components_
        # find k features in data that has the largest relative contribution to explained variance in X values
        k = 100
        top_contributions, top_contributors = get_top_k_contributions_and_contributors(k=k, all_pcs=all_pcs)
        # make_fig_contributions(top_contributors, top_contributions, k)
        # make_fig_heatmap(X_train[top_contributors])
        # Find pcs which explaining (contributing to explained var) over 90 and 99% of variance
        ratios = report_pca.explained_variance_ratio_
        index_90 = get_pcs_num_explains_p_ratio_of_var(0.99, ratios)
        index_99 = get_pcs_num_explains_p_ratio_of_var(0.9, ratios)
        print(f"In X_train, {index_90 + 1} principal components explain over 99% of variance\n")
        print(f"In X_train, {index_99 + 1} principal components explain over 90% of variance\n")
        print(f"**************************************************\n"
              f"Distribution of categorical variables in data:"
              f"**************************************************\n"
              "Gender:\n"
              f"X['gender'].value_counts(): \n{X['gender'].value_counts()}\n"
              f"X_train['gender'].value_counts():\n{X_train['gender'].value_counts()}\n"
              f"X_test['gender'].value_counts():\n{X_test['gender'].value_counts()}\n"
              f"\n**************************************************\n"
              "Treatment_group (coil):\n"
              f"X['Treatment_group'].value_counts():\n {X['Treatment_group'].value_counts()}\n"
              f"X_train['Treatment_group'].value_counts():\n{X_train['Treatment_group'].value_counts()}\n"
              f"X_test['Treatment_group'].value_counts():\n{X_test['Treatment_group'].value_counts()}\n"
              "Response to treatment:\n"
              f"\n**************************************************\n"
              f"['6-weeks_HDRS21_class'].value_counts():\n{y['6-weeks_HDRS21_class'].value_counts()}\n"
              f"y_train['6-weeks_HDRS21_class'].value_counts():\n{y_train['6-weeks_HDRS21_class'].value_counts()}\n"
              f"y_test['6-weeks_HDRS21_class'].value_counts():\n{y_test['6-weeks_HDRS21_class'].value_counts()}\n"
              f"\n**************************************************\n"
              f"Principal components:\n")

        k = 100
        selector = SelectKBest(k=k)
        X_new = selector.fit_transform(X_train, y_train)
        selected_indices = selector.get_support(indices=True)
        feature_scores = selector.scores_
        best_k_features = list(X_train.iloc[:, list(selected_indices)].columns)
        # print("Selected indices:", selected_indices)
        print(f"K = {k}  best of X train (SelectKbest)")
        print(f"On X train:\n{k} Feature scores:\n")
        for i in range(len(best_k_features)):
            print(f"i = {i}")
            print("feature ", best_k_features[i])
            print("score (f_classif) ", feature_scores[i])
            print("p_val ", selector.pvalues_[i])

    if args['classification']:
        if args['lite_mode']:  # just for debugging. using one small grid
            # param_pipe_list = [[param2a,pipe_smote_2a]]
            # param_pipe_list = [[param3a, pipe_smote_3a]] # CHECKED
            # param_pipe_list = [[param5a, pipe_smote_5a]]
            # param_pipe_list = [[param5a, pipe5a]]
            # param_pipe_list = [[param5a,pipe_smote_5a]]

            # param_pipe_list = [[param5b,pipe_smote_5b]]

            # param_pipe_list = [[param5, pipe5]]

            # param_pipe_list = [[param7a,pipe7a]]
            # param_pipe_list = [[param7a,pipe_smote_7a]]
            # param_pipe_list = [[param_6_classifier_only,pipe6_classifier_only  ]]
            # param_pipe_list = [[param6a, pipe_smote_6a]] # CHECKED
            # param_pipe_list = [[param7a, pipe_smote_7a]] # CATBOOST
            # param_pipe_list = [[param8a, pipe_smote_8a]]  # CHECKED
            # param_pipe_list = [[param8, pipe8_classifier_only]]
            # param_pipe_list = [[param3b, pipe_smote_3b]] # CHECKED
            # param_pipe_list = [[param6b, pipe_smote_6b]] # CHECKED
            # param_pipe_list = [[param7, pipe7]] # CHECKED
            param_pipe_list = [[param7b, pipe7b]] # CHECKED

            # param_pipe_list = [[param7b, pipe_smote_7b]] # CATBOOST
            # param_pipe_list = [[param_2_classifier_only, pipe2_classifier_only]]
            # param_pipe_list = [[param_3_classifier_only, pipe3_classifier_only]]
            # param_pipe_list = [[param_6_classifier_only, pipe6_classifier_only]]
            # param_pipe_list = [[param_8_classifier_only, pipe8_classifier_only]]
            # param_pipe_list = [[param2a_smote_no_dimension_red, pipe_smote_2a_no_dimension_reduction]]
            # param_pipe_list = [[param6_smote_no_dimension_red, pipe_smote_6_smote_no_dimension_red]]
            # param_pipe_list = [[param_8_dimension_red_only,pipe8_dimension_red_only]]

        else:  # more than one model
            # pipe is represent the steps we want to execute, param represents which args we want to execute with
            param_pipe_list = []  # put all the pipe and param pairs you want

    for pair in param_pipe_list:
        total_folds = [0]  # counter
        all_splits_yts, all_splits_yps = [], []  # all_splits_yts and all_splits_yps are lists of vectors, one vector for each split (n_iter * n_splits len).
        param = pair[0]
        pipe = pair[1]
        choice_scores = [0 for i in range(args['cv'])]  # list of scores in length num of folds
        best_score_by_now = [0]
        if args['exhaustive_grid_search']:
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

        # REPORT 1 -
        # Print report about cross-validation (selection of the best hyper-parameters)

        search.fit(X_train.astype(float), y_train)
        best_cv_iter_idx = search.best_index_  # index of the iteration in cross val search which had best parsms (0<=best_cv_iter_idx <=niter)
        best_cv_iter_first_split_idx = best_cv_iter_idx * n_splits
        best_cv_iter_all_splits_indices = range(best_cv_iter_first_split_idx,
                                                best_cv_iter_first_split_idx + n_splits)  # the exact range of the 5 test scores of the best index configuration
        best_cv_iter_yps_list = [all_splits_yps[i] for i in best_cv_iter_all_splits_indices]
        best_cv_iter_yps_list_ndarray = np.concatenate(best_cv_iter_yps_list)  # turn to nd_array
        best_cv_iter_yts_list = [all_splits_yts[index] for index in best_cv_iter_all_splits_indices]
        best_cv_iter_yts_list_ndarray = np.concatenate(best_cv_iter_yts_list)

        print_conclusions(X_train, pipe, search, y_pred=best_cv_iter_yps_list_ndarray,
                          y_true=best_cv_iter_yts_list_ndarray, mode='cv', folder=folder)

        # REPORT 2 -
        # Print report about trained model after refit on Xtrain and prediction on y_train
        y_pred_on_train_data = search.predict(X_train)
        print_conclusions(X_train, pipe, search, y_pred=y_pred_on_train_data, y_true=y_train, mode='train',
                          folder=folder)

        # REPORT 3 -
        # Print report about final prediction on test set (Xtest)
        y_pred_on_test_data = search.predict(X_test)
        print_conclusions(search=search, y_true=y_test, y_pred=y_pred_on_test_data, mode='test', folder=folder)

        ci_metrics = ['precision','accuracy','f1']
        for metric in ci_metrics:
            ci_model, ci_dummy   = ci(X_test,y_test,search,metric)
            print(f"CIs of {metric} - generated by bootstrapping from test set:\nTrained Model CI: {ci_model} Dummy (baseline) Model CI (Always predicts 1): {ci_dummy}")
            print("Intersection <=> models are significantly different")

print(f"<<<<<<<<<<<<<<<<<<<<< GSCVrunner.py finished successfuly<<<<<<<<<<<<<<<<<<<<<")
if args['stdout_to_file']:
    log_file.close()
