from keras.layers import Input,Dense
from keras.models import Model
import numpy as np
import pandas as pd

#prepare data
tb_path=r'F:\chenyw\毕业论文\毕业\data\gz_con.xlsx'
tb= pd.read_excel(tb_path ,'Sheet1')
x=tb.iloc[:,1:].values
print(tb.shape)
print(x[:4])


#make encoder
input_data=Input((,6))
encoded=Dense(256,activation='relu')(input_data)
encoded=Dense(128,activation='relu')(encoded)
encoded=Dense(64,activation='relu')(encoded)
encoded=Dense(32,activation='relu')(encoded)
encoded_data=Dense(2)(encoded)

#make decoder
decoded=Dense(32,activation='relu')(encoded_data)
decoded=Dense(64,activation='relu')(decoded)
decoded=Dense(128,activation='relu')(decoded)
decoded_data=Dense(256,activation='relu')(decoded)

#make model