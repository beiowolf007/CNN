from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.layers.normalization import BatchNormalization
import numpy
import pandas as pd
from sklearn import preprocessing
from sklearn import model_selection



# prepare data
tb_path = r'E:\chenyw\毕业论文\湖南\ssss.xlsx'
tb = pd.read_excel(tb_path, sheet_name='ep')

x = tb.iloc[:, 0:8].values
y = tb.iloc[:, -1].values
x = x.reshape(x.shape[0], 8, 1).astype('float32')
print(y.shape)
t = y.reshape(-1, 1)
t = preprocessing.OneHotEncoder(sparse=False).fit_transform(t).toarray()
# x_train, y_train, x_test, y_test = model_selection.train_test_split(x, t, test_size=0.25)
print(y[0])
print(t[0])

# make net
# model=Sequential()
# # model.add(BatchNormalization(input_shape=(None,8)))
# model.add(Conv1D(filters=64,kernel_size=3,strides=1,input_shape=(8,1),
#                  padding='same',activation='relu'))
# model.add(MaxPooling1D(pool_size=2,padding='same'))
# model.add(BatchNormalization(axis=1))
# model.add(Dropout(0.5))
# model.add(Conv1D(filters=128,kernel_size=3,padding='same',activation='relu'))
# model.add(MaxPooling1D(pool_size=2))
# # model.add(BatchNormalization(axis=1))
# # model.add(Dropout(0.5))
# # model.add(Conv1D(filters=256,kernel_size=3,padding='same',activation='relu'))
# # model.add(MaxPooling1D(pool_size=2))
# # model.add(Dropout(0.5))
# model.add(Flatten())
# model.add(Dense(128,activation='relu'))
# model.add(Dense(64,activation='relu'))
# model.add(Dense(32,activation='relu'))
# model.add(Dense(5,activation='softmax'))
# model.compile(loss='categorical_crossentropy',optimizer='Adam')
# metrics=['accuracy']
#
# #trainning net
# model.fit(x_train,y_train,validation_data=(x_test,y_test),epochs=2,batch_size=64)
# scores=model.evaluate(x_test,y_test,verbose=0)
# print(scores)
