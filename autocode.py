from keras.layers import Input,Dense
from keras.models import Model
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.cluster import AgglomerativeClustering

#prepare data
tb_path=r'F:\chenyw\毕业论文\毕业\data\gz_con.xlsx'
tb= pd.read_excel(tb_path ,'Sheet1')
tb2=tb.iloc[:,2:]
tb2=tb2.apply(lambda x:(x-np.min(x))/(np.max(x)-np.min(x)))
x=tb2.iloc[:,:].values
print(tb2.head())
print(x[:4])


#make encoder
input_data=Input(shape=(6,))
encoded=Dense(128,activation='relu')(input_data)
encoded=Dense(64,activation='relu')(encoded)
encoded=Dense(10,activation='relu')(encoded)
encoded_data=Dense(2)(encoded)

#make decoder
decoded=Dense(10,activation='relu')(encoded_data)
decoded=Dense(64,activation='relu')(decoded)
decoded=Dense(128,activation='relu')(decoded)
decoded_data=Dense(6,activation='tanh')(decoded)

#make model
autoencoder=Model(inputs=input_data,outputs=decoded_data)

#make encoder
encoder_Model = Model(inputs=input_data,outputs=encoded_data)

#compile and train
autoencoder.compile(optimizer='Adam',loss='mse')
autoencoder.fit(x,x,batch_size=15,epochs=30,shuffle=True)

#to cluster
cluster_data=encoder_Model.predict(x)
end_data=autoencoder.predict(x)
print(end_data[:3,])
print(cluster_data[0:3,])
cluster_alg=AgglomerativeClustering(n_clusters=4)
cluster_alg.fit(cluster_data)
print(cluster_data[:3,])
