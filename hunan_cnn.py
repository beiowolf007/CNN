import cntk
import keras
from keras import Model
import numpy as np
import pandas as pd
import sklearn as sl
from sklearn import preprocessing
from sklearn import model_selection



# prepare data
tb_path = r'E:\chenyw\毕业论文\湖南\ssss.xlsx'
tb = pd.read_excel(tb_path, sheet_name='ep')

x = tb.iloc[:, 0:8].values
y = tb.iloc[:, -1].values
t = y.reshape(-1, 1)
ohe = preprocessing.OneHotEncoder()
ohe.fit(t)
x_train, y_train, x_test, y_test = model_selection.train_test_split(x, y, test_size=0.25)
print(x[:5], t[:5])

# make net
