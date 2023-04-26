import pandas as pd
import sklearn
n_components = [1,2,3,4,5]
df = pd.read_csv('all_data.csv')
scaler = sklearn.preprocessing.StandardScaler()
df = scaler.fit_transform(df)
for n in n_components:
    pca = sklearn.decomposition.PCA(n)
    _ = pca.fit(df)
    W = pca.components_.T
    print(W.shape)