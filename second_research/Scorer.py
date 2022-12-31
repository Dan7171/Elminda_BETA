def CV_Score(y_true,y_pred):
    global yt,yp
    yt.append(y_true)
    yp.append(y_pred)
    cvscore = f1_score(y_true, y_pred)
    return cvscore

def scorer():
    return make_scorer(CV_Score)
	
	
def print_precission_and_recall(search,cv)

    yp = []
    yt = []


    gs = GridSearchCV(estimator=pipeline, param_grid=param, scoring=scorer(), cv=cv).fit(X_train, y_train)

    # collect every split into a full array
    best_ind = search.best_index_
    cnt_splits = cv.cvargs['n_splits']
    chooseThese = range(best_ind*cnt_splits,best_ind*cnt_splits+cnt_splits,1)
    del yp[1::2]
    yp_best = [yp[index] for index in chooseThese]
    yp_cv = np.concatenate(yp_best)
    del yt[1::2]
    yt_best = [yt[index] for index in chooseThese]
    yt_cv = np.concatenate(yt_best)
    cm = confusion_matrix(yt_cv, yp_cv)
    print(cm)
    # plt.figure()
    # fig = metrics.ConfusionMatrixDisplay.from_predictions(yt_cv, yp_cv)
    print("Response rate:",y_train.values.mean())
    print("CV Precision:", cm[1][1] / (cm[1][1] + cm[0][1]))
    print("CV Recall:", cm[1][1] / (cm[1][1] + cm[1][0]))
    

 

# from sklearn.model_selection import RandomizedSearchCV, cross_validate

# # Define the model and the parameter grid
# param_grid = {'param1': [1, 2, 3], 'param2': [4, 5, 6]}

# # Create the RandomizedSearchCV object with the parameter grid and the splitter
# random_search = RandomizedSearchCV(pipe1a, param1a, cv=10, scoring=['precision', 'recall', 'accuracy'])

# # Fit the model to the data
# random_search.fit(X_train,y_train.ravel())

# # Get the best model from the search
# best_model = random_search.best_estimator_

# # Get the cross-validation scores
# scores = random_search.cv_results_

# avg_precision = scores['mean_test_precision'].mean()
# avg_recall = scores['mean_test_recall'].mean()
# avg_accuracy = scores['mean_test_accuracy'].mean()