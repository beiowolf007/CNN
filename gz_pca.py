import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import pandas as pd

tb_path=r'F:\chenyw\毕业论文\毕业\data\gz_con.xlsx'
tb= pd.read_excel(tb_path ,'Sheet1')
x=tb.iloc[:,1:].values

my_pca=PCA(n_components='mle',svd_solver='full')
my_pca.fit(x)
print(my_pca.components_)
print(my_pca.explained_variance_ratio_ )