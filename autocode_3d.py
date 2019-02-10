from keras.layers import Input,Dense
from keras.models import Model
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from keras import regularizers
from sklearn.cluster import KMeans

def make_list(data,label):
    list0 = list()
    list1 = list()
    list2 = list()
    list3 = list()
    list4=list()
    list5=list()
    for i in range(len(data)):
        if label[i]==0:
            templist=[data[i,0],data[i,1],data[i,2],i,0]
            list0.append(templist)
        elif label[i]==1:
            templist = [data[i, 0], data[i, 1],data[i,2],i,1]
            list1.append(templist)
        elif label[i]==2:
            templist = [data[i, 0], data[i, 1],data[i,2],i,2]
            list2.append(templist)
        elif label[i]==3:
            templist = [data[i, 0], data[i, 1],data[i,2],i,3]
            list3.append(templist)
        elif label[i]==4:
            templist = [data[i, 0], data[i, 1],data[i,2],i,4]
            list3.append(templist)
        elif label[i]==5:
            templist = [data[i, 0], data[i, 1],data[i,2],i,5]
            list3.append(templist)
    array0=np.array(list0)
    array1=np.array(list1)
    array2 = np.array(list2)
    array3 = np.array(list3)
    # array4=np.array(list4)
    return array0,array1,array2,array3
    

#prepare data
tb_path=r"F:\work\毕业论文\毕业\data\gz_cx_con.xlsx"
tb= pd.read_excel(tb_path ,'Sheet2')
tb2=tb.iloc[:,2:8]
# tb2=tb2.apply(lambda x:(x-np.min(x))/(np.max(x)-np.min(x)))
# x=tb2.iloc[:,:].values
x=tb2.iloc[:,:].values
# x=pp.minmax_scale(t)
# x=pp.normalize(t,axis=0)
print(tb2.shape)
print(tb2.head())
print(x[:4])



#make encoder
input_data=Input(shape=(6,))
encoded=Dense(128,activation='relu',activity_regularizer=regularizers.l1(10e-5))(input_data)
encoded=Dense(64,activation='relu',activity_regularizer=regularizers.l1(10e-5))(encoded)
encoded=Dense(10,activation='relu',activity_regularizer=regularizers.l1(10e-5))(encoded)
encoded_data=Dense(3)(encoded)

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
autoencoder.fit(x,x,batch_size=15,epochs=200,shuffle=True)

#to encode
cluster_data=encoder_Model.predict(x)
end_data=autoencoder.predict(x)
# print(end_data[:3,])
# print(cluster_data[0:3,])

#cluster and show up
mark_dic={0:'o',1:'v',2:'*',3:'p'}


fig=plt.figure()
ax1=Axes3D(fig)
cluster_alg=KMeans(n_clusters=4,n_jobs=-1)
clusted=cluster_alg.fit(cluster_data)
# ax1.scatter3D(cluster_data[:,0],cluster_data[:,1],cluster_data[:,2],c=clusted.labels_)
list0,list1,list2,list3=make_list(cluster_data,clusted.labels_)
ax1.scatter3D(list0[:, 0], list0[:, 1],list0[:,2],marker='$A$', c='b')
ax1.scatter3D(list1[:, 0], list1[:, 1],list1[:,2],marker='$B$', c='g')
ax1.scatter3D(list2[:, 0], list2[:, 1],list2[:,2],marker='$C$', c='r')
ax1.scatter3D(list3[:, 0], list3[:, 1],list3[:,2],marker='$D$', c='c')

outpath=r'F:\work\毕业论文\毕业\data\clust\gz_clust3d_2.xlsx'
outlist=np.ones([1,5])
outlist=np.row_stack([list0,list1,list2,list3])
outTable=pd.DataFrame(outlist,columns=['X','Y','Z','序号','分类'])
outTable.to_excel(outpath,sheet_name='clu3d')

plt.show()

