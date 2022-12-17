

#RUN ARGUMENTS
args = {
"rs" : 42 ,#random state
"X_version" :  2 ,# 1 = no dimension reduction (1000 features), 2 = medium dimension reduction (500 features), 3 = significant reduction (100-200 features)
"split_rows" : 1 , # 1 = don't split rows, 2 = split to h1 and h7
"drop_out_correlated" : True ,
"age_under_50" : True ,
"debug" : False ,
"classification_type": "normal", # "normal" =2 classes:  <50% , >50% change, "extreme" = 3 classes:  <30% , 30-70% , >70% change
"scoring": "all_scores" # "accuracy" / "all_scores"
}