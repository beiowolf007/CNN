from keras.layers import Input,Dense
from keras.models import Model
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.cluster import AgglomerativeClustering
import openpyxl

def make_list(data,label,phase):
    list0 = list()
    list1 = list()
    list2 = list()
    list3 = list()
    for i in range(len(data)):
        if label[i]==0:
            templist=[data[i,0],data[i,1],i,0,phase]
            list0.append(templist)
        elif label[i]==1:
            templist = [data[i, 0], data[i, 1],i,1,phase]
            list1.append(templist)
        elif label[i]==2:
            templist = [data[i, 0], data[i, 1],i,2,phase]
            list2.append(templist)
        elif label[i]==3:
            templist = [data[i, 0], data[i, 1],i,3,phase]
            list3.append(templist)
    array0=np.array(list0)
    array1=np.array(list1)
    array2 = np.array(list2)
    array3 = np.array(list3)
    return array0,array1,array2,array3
    

#prepare data
tb_path=r"F:\chenyw\毕业论文\毕业\data\gz_con.xlsx"
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


#to encode
# cluster_data=encoder_Model.predict(x)
# end_data=autoencoder.predict(x)
# print(end_data[:3,])
# print(cluster_data[0:3,])

#cluster and show up
plot_num=331
mark_dic={0:'o',1:'v',2:'*',3:'P'}
# outlist_f=np.ones([1,5])
for i in range(0,9):
    autoencoder.fit(x, x, batch_size=15, epochs=70, shuffle=True)
    cluster_data = encoder_Model.predict(x)
    end_data = autoencoder.predict(x)
    cluster_alg = AgglomerativeClustering(n_clusters=4, linkage='ward')
    clusted = cluster_alg.fit(cluster_data)
    list0, list1, list2, list3 = make_list(cluster_data, clusted.labels_,i)
    plt.subplot(plot_num)
    plt.scatter(list0[:, 0], list0[:, 1], marker=mark_dic[0])
    plt.scatter(list1[:, 0], list1[:, 1], marker=mark_dic[1])
    plt.scatter(list2[:, 0], list2[:, 1], marker=mark_dic[2])
    plt.scatter(list3[:, 0], list3[:, 1], marker=mark_dic[3])
    plt.title("ward clust  %d times" % i)
    plot_num = plot_num + 1
    outpath=r'F:\chenyw\毕业论文\毕业\data\clust\gz_clust%d.xlsx'%i
    outlist=np.row_stack([list0,list1,list2,list3])
    # outlist_f=np.row_stack([outlist_f,outlist])
    outdf=pd.DataFrame(outlist,columns=['x','y','序号','分类','运行次数'])
    outdf.to_excel(outpath,sheet_name='clust')
# outpath=r'F:\chenyw\毕业论文\毕业\data\clust\gz_con_clust.xlsx'


plt.tight_layout()
plt.show()

