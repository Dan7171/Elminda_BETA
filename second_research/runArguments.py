

#RUN ARGUMENTS
args = {
"rs" : 42 ,#random state
"X_version" :  1 ,# 1 = no dimension reduction (1000 features), 2 = medium dimension reduction (500 features), 3 = significant reduction (100-200 features)
"split_rows" : 1 , # 1 = don't split rows, 2 = split to h1 and h7
"drop_out_correlated" : True ,
"age_under_50" : False ,
"debug" : False ,
"classification_type": "normal", # "normal" =2 classes:  <50% , >50% change /"extreme" = 3 classes:  <30% , 30-70% , >70% change
"scoring_method": "accuracy", # "accuracy" / "precision"/"recall'/"all_scores".
"both": False, # True- train on both research 1 and research 2 data , False- train on research 2 only.IMPORTANT: if set to True, use X_version = 1 only!!!
"cv":4, # param for cv
"n_iter": 50 # param for cv
}