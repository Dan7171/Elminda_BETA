

#RUN ARGUMENTS
args = {
"rs" : 42 ,#random state
"X_version" :  1 ,# 1 = no dimension reduction (1000 features)/ 2 = medium dimension reduction (500 features), 3 = significant reduction (100-200 features)
"split_rows" : 'h7' , # 'normal' = normal train test split (dont drop anything) / 'h1'- drop rows where Treatment != 'h1' and then use train test split.
# / 'h7' =  drop all rows where Treatment != 'h7' and then use train test split.
# 'h1h7', take both 'h1' and 'h7' options
"drop_out_correlated" : False , # True: use corellationDropper transformer in pipeline, False: do not use it
"age_under_50" : False , # use only subjects with age less than 50
"debug" : False ,# never use
"exhaustive_grid_search": False,
"classification_type": "normal", # "normal" =2 classes:  <50% , >50% change / "extreme" = 3 classes:  <30% , 30-70% , >70% change
"scoring_method": 'accuracy', # "accuracy" / "precision"/"recall'/'custom_f1_scorer' [elminda_scorer is offir's f1 that outputs cross val's precision and recall].
"both": False, # True- train on both research 1 and research 2 data , False- train on research 2 only.IMPORTANT: if set to True, use X_version = 1 only!!!
"cv":5, # param for cv
"n_iter": 80, # param for randomized cv 
"n_jobs":1, # num of threads each model is generating to speed up grid search,
"use_gamma_columns":True #True: using Gamma columns. False: not using them. IMPORTANT: for True, use with X_version = 1 only
}