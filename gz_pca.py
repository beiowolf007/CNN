import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import pandas as pd

tb_path=r'F:\chenyw\毕业论文\毕业\data\gz_con.xlsx'
for i in range(0,4):
    sheet_name='c%d'%i
    tb= pd.read_excel(tb_path ,sheet_name)
    x=tb.iloc[:,2:8].values
    # print(x[:3])
    my_pca=PCA(n_components='mle',svd_solver='full')
    my_pca.fit(x)
    print('The PCA for cluster %d'%i)
    print(my_pca.components_)
    print(my_pca.explained_variance_ratio_ )