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
#import models_runner_r1

def select_best_model(X_train, y_train)->tuple:
    """Given X and y (dfs), returning  a 2-sized tuple (model,score) where model is the model name and score is an accuracy score,
    out of a dictionary models_to_scores """
    models_to_scores = get_model_names_to_model_accuracy_scores_dict()
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


bna_path = 'second_research\EYEC_Cordance_Gamma.csv'
clinical_path = 'second_research\BW_clinical_data.csv'
# Step 1: prepare X for model training
X = numeric_df_initializer.generate_prediction_df(bna_path,clinical_path) #initial numeric-predicting data frame
y_name = 'class_week_6'
print(X.head)
#X.to_csv('second_research/X_before prediction.csv',index = False)
y = get_y_column(X,y_name) # implement
X_train, X_test,y_train, y_test = train_test_split(X, y, test_size=0.2,random_state = 0) #rs is seed


# Step 2: select classification model with k fold cross validation  

X_train_copy = X_train.copy()  
y_train_copy = y_train.copy()
selected_model = select_best_model(X_train_copy, y_train_copy) # selecting best classification model with cross validation. using 80% of data
# one shot preditciton:
selected_model.fit(X_train,y_train)
y_pred = selected_model.predict(X_test) # real 1 shot prediction
selected_model_accuracy_score = metrics.accuracy_score(y_test,y_pred)


