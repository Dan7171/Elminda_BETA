from matplotlib import pyplot as plt
import numeric_df_initializer
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold 
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
import warnings
import main_caller_r2
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore')
warnings.simplefilter(action='ignore')
X,y = main_caller_r2.get_X_y(y_name='6-weeks_HDRS21_class')
# credits:
#https://scikit-learn.org/stable/tutorial/statistical_inference/putting_together.html#open-problem-stock-market-structure

pca = PCA()
scaler = StandardScaler()
#add selectKbest to pipeline
kBest_selector = SelectKBest()
#print(X) #v
#print(y) #v
X_train, X_test,y_train, y_test = train_test_split(X, y, test_size=0.2,random_state = 0)  
# Parameters of pipelines can be set using '__' separated parameter names:



clf1 = LogisticRegression(max_iter=10000)
param1 = {
    "pca__n_components": [5, 15, 30, 45, 60],
    "logistic__C": np.logspace(-4, 4, 4),
    "kBest__k": [5,10,15],
    "kBest__score_func" : ['mutual_info_classif','f_classif']    
}
pipe1 = Pipeline(steps=[("scaler", scaler), ("pca", pca), ("kBest",kBest_selector),("logistic", clf1)])
search1 = GridSearchCV(pipe1, param1, n_jobs=2,refit=True).fit(X_train, y_train) #grdidsearcg cv finds a pipeline with the best configuration of hyperparameters for the specific pipeline
print(search1)
print("after fit(x_train,y_train):")
print("BEST PARAMS: ",search1.best_params_)
print("BEST SCORE: ", search1.best_score_)


clf1 = LogisticRegression(max_iter=10000)


# # Plot the PCA spectrum
# pca.fit(X)

# fig, (ax0, ax1) = plt.subplots(nrows=2, sharex=True, figsize=(6, 6))
# ax0.plot(
#     np.arange(1, pca.n_components_ + 1), pca.explained_variance_ratio_, "+", linewidth=2
# )
# ax0.set_ylabel("PCA explained variance ratio")

# ax0.axvline(
#     search.best_estimator_.named_steps["pca"].n_components,
#     linestyle=":",
#     label="n_components chosen",
# )
# plt.show()