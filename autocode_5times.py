from keras.layers import Input,Dense
from keras.models import Model
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from keras import regularizers

def make_list(data,label):
    list0 = list()
    list1 = list()
    list2 = list()
    list3 = list()
    list4=list()
    for i in range(len(data)):
        if label[i]==0:
            templist=[data[i,0],data[i,1],i,0]
            list0.append(templist)
        elif label[i]==1:
            templist = [data[i, 0], data[i, 1],i,1]
            list1.append(templist)
        elif label[i]==2:
            templist = [data[i, 0], data[i, 1],i,2]
            list2.append(templist)
        elif label[i]==3:
            templist = [data[i, 0], data[i, 1],i,3]
            list3.append(templist)
        elif label[i] ==4:
            templist = [data[i, 0], data[i, 1], i,4]
            list4.append(templist)
    array0=np.array(list0)
    array1=np.array(list1)
    array2 = np.array(list2)
    array3 = np.array(list3)
    # array4=np.array(list4)
    return array0,array1,array2,array3
    

#prepare data
tb_path=r"F:\work\毕业论文\毕业\data\gz_cx_con.xlsx"
tb= pd.read_excel(tb_path ,'ad')
tb2=tb.iloc[:,2:8]
# tb2=tb2.apply(lambda x:(x-np.min(x))/(np.max(x)-np.min(x)))
x=tb2.iloc[:,:].values
print(tb2.head())
print(x[:4])


#make encoder
input_data=Input(shape=(6,))
encoded=Dense(128,activation='relu',activity_regularizer=regularizers.l1(10e-5))(input_data)
encoded=Dense(64,activation='relu',activity_regularizer=regularizers.l1(10e-5))(encoded)
encoded=Dense(10,activation='relu',activity_regularizer=regularizers.l1(10e-5))(encoded)
encoded_data=Dense(2)(encoded)
# encoded_data=Dense(2,activity_regularizer=regularizers.l1(10e-5))(encoded)

#make decoder
decoded=Dense(10,activation='relu')(encoded_data)
decoded=Dense(64,activation='relu')(decoded)
decoded=Dense(128,activation='relu')(decoded)
decoded_data=Dense(6,activation='linear')(decoded)

#make model
autoencoder=Model(inputs=input_data,outputs=decoded_data)

#make encoder
encoder_Model = Model(inputs=input_data,outputs=encoded_data)

#compile and train
autoencoder.compile(optimizer='Adam',loss='mse')


#to encode
# cluster_data=encoder_Model.predict(x)
# end_data=autoencoder.predict(x)
# print(end_data[:3,])
# print(cluster_data[0:3,])

#cluster and show up
plot_num=331
mark_dic={0:'o',1:'v',2:'*',3:'P'}
# outlist_f=np.ones([1,5])
autoencoder.fit(x, x, batch_size=15, epochs=2, shuffle=True)
cluster_data = encoder_Model.predict(x)
end_data = autoencoder.predict(x)
cluster_alg = KMeans(n_clusters=4,n_jobs=-1)
clusted = cluster_alg.fit(cluster_data)
list0, list1, list2,list3= make_list(cluster_data, clusted.labels_)
# plt.scatter(cluster_data[:,0],cluster_data[:,1],c=clusted.labels_)
# plt.subplot(plot_num)
plt.scatter(list0[:, 0], list0[:, 1],marker='$A$', c=clusted.labels_)
plt.scatter(list1[:, 0], list1[:, 1],marker='$B$', c=clusted.labels_)
plt.scatter(list2[:, 0], list2[:, 1],marker='$C$', c=clusted.labels_)
plt.scatter(list3[:, 0], list3[:, 1],marker='$D$', c=clusted.labels_)
# plt.scatter(list4[:, 0], list4[:, 1], c=clusted.labels_)
outpath=r'F:\work\毕业论文\毕业\data\clust\gz_clust.xlsx'
outlist=np.row_stack([list0,list1,list2,list3])
# outlist_f=np.row_stack([outlist_f,outlist])
outdf=pd.DataFrame(outlist,columns=['x','y','序号','分类'])
outdf.to_excel(outpath,sheet_name='clust')
# outpath=r'F:\chenyw\毕业论文\毕业\data\clust\gz_con_clust.xlsx'


plt.tight_layout()
plt.show()

