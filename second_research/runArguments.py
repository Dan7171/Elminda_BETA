

#RUN ARGUMENTS
args = {
"rs" : 42 ,#random state

"X_version" :  1 ,# 1 = no dimension reduction (1000 features) / 2 = medium dimension reduction (500 features), 3 = significant reduction (100-200 features)

"split_rows" : 'normal' , # 'normal' = normal train test split (dont drop anything) / 'h1'- drop rows where Treatment != 'h1' and then use train test split.
# / 'h7' =  drop all rows where Treatment != 'h7' and then use train test split /
# 'h1h7', take both 'h1' and 'h7' options

"drop_out_correlated" : False , # DO NOT CHANGE TO TRUE  True: use corellationDropper transformer in pipeline, False: do not use it

"age_under_50" : False , # use only subjects with age less than 50

"debug" : False ,# NEVER USE

"exhaustive_grid_search": False, # or randomized

"classification_type": "normal", # "normal" = 2 classes: < 50% change, => 50% change / "extreme" = 3 classes:  <30% , 30-70% , >70% change

"scoring_method": 'accuracy', # "accuracy" /'f1'/ 'roc_auc' /'precision' /'recall' (sklearn metrics score funcs)

"both": True, # (for now works on classificatin only) True- train on both research 1 and research 2 data , False- train on research 2 only.IMPORTANT: if set to True, use X_version = 1 only!!!

"cv":8, # param for cv

"balance_y_values" : True
    , # working for clasification only. balancing the number of responsive and non responsive (y categories)

"n_iter": 1000, # param for randomized cv - num of combinations to try in randomized search

"n_jobs":1, # num of threads each model is generating to speed up grid search,

"use_gamma_columns":True, # True: using Gamma columns. False: not using them. IMPORTANT: for True, use with X_version = 1 only

"classification": True, # true - classification cv train, false- regression cv train

"lite_mode": True, # True for running search on one model only, False for running on more models (one after the next)

"test_size": 0.1 # train test splits test size

}
if args["both"] or args["balance_y_values"]:
    args['classification'] = True

if not args['classification']:
    args["balance_y_values"] = False
    args["both"] = False



# todo
# 1. regressors?
# 2. normaliztion?

# normalize a series (checked)

    # from sklearn.preprocessing import MinMaxScaler
    # scaler = MinMaxScaler()
    # # Normalize the Series using MinMaxScaler
    # normalized_s = scaler.fit_transform(df.values.reshape(-1, 1))
    # # Convert the normalized values back to a Series
    # normalized_s = pd.Series(normalized_s.reshape(-1))
    #

# # normalize a df
    # import pandas as pd
    # from sklearn import preprocessing
    #
    # x = df.values #returns a numpy array
    # min_max_scaler = preprocessing.MinMaxScaler()
    # x_scaled = min_max_scaler.fit_transform(x)
    # df = pd.DataFrame(x_scaled)
