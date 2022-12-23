import pandas as pd

"""The goal of this module is to take a specific version of research 1 and to suit it to the shape of Xy in research 2"""




def get_y_column(X,y_name):
    """Given a df X and a string y_name, returning a column y of it, corresponding to X rows"""

    if y_name == '6-weeks_HDRS21_class': 
        base = 'Baseline_HDRS21_totalscore'
        end = '6-weeks_HDRS21_totalscore'     
        X = X[X[end].notna()] 

    if y_name == '6-weeks_HARS_class':
        base = ' Baseline_HARS_totalscore'
        end = '6-weeks HARS_totalscore'  
    
    change = '%_change_rate'
    X[change] = ((X[end] - X[base]) / X[base])* 100  # increase in % in  total score 
    if args["debug"]:
        print_X = X[['subject',base,end,change,'age']]
        print(print_X)
        print_X.to_csv('second_research\output_csvs\debug_file.csv',index = False)  
    y = pd.DataFrame()

    y[y_name] = X.apply(lambda subject: convert_change_rate_to_class(subject[change]),axis=1) #apply function using more than one column.https://stackoverflow.com/questions/13331698/how-to-apply-a-function-to-two-columns-of-pandas-dataframe
       
    #print("counts of y's y_name column" 
    #print(y[y_name].value_counts())
    X.drop(change,axis=1,inplace=True) # 1650 cols to 1649. remove col from X since it was just a helpfer col to get out y col (and can effect prediction negatively)
    return y

path= "first_research\X_(all_features)_beer_yaakov_same_pattern_as_reseatch2.csv"
df = pd.read_csv(path)
