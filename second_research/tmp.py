import pandas as pd
import main_caller_r2, numeric_df_initializer
path_Xy_r1 = "first_research\X_(all_features)_beer_yaakov_same_pattern_as_reseatch2.csv"        
data_r1 = pd.read_csv(path_Xy_r1)
print(data_r1)
y = main_caller_r2.get_y_column(data_r1,'6-weeks_HDRS21_class')
to_drop = ['Baseline_HDRS21_totalscore','6-weeks_HDRS21_totalscore']
X = data_r1.drop(columns=to_drop)
print(X)
print(y)
Xy_r1 =(X.join(y))
print(Xy_r1)
Xy_r1.to_csv('first_research\Xy_at_r2_format.csv',index=False)
print(pd.read_csv('first_research\Xy_at_r2_format.csv').shape)