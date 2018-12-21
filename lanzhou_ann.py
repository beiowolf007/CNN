from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras.utils import plot_model
from keras.layers import convolutional as cv
from keras.layers import pooling as pl
import keras
import pandas as pd
from sklearn import preprocessing
from sklearn import model_selection

# prepare data
tb_path = r'f:\chenyw\毕业论文\甘肃\lanzhou.xls'
tb = pd.read_excel(tb_path, sheet_name='lanzhou')
tb2 = tb
# tb2=tb[tb['dd']==5]
print(tb2.shape)
x = tb2.iloc[:, 0:8].values
y = tb2.iloc[:, 9].values
#x = x.reshape((1,8),1,8, 1).astype('float32')
# y=y.reshape(len(y),-1).astype('float32')
x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.25)
#print(x.head())
print(x_train.shape)
x_train=x_train.reshape(381,8,1)
x_test=x_test.reshape(128,8,1)
print(x_train[0])
print(y[:3])

# make net
model = Sequential()
#model.add(BatchNormalization(input_shape=(8,)))
model.add(cv.Conv1D(filters=10,kernel_size=2,padding='same',input_shape=(8,1),activation='relu'))

model.add(pl.MaxPooling1D(pool_size=2,strides=None,padding='valid'))
model.add(Dropout(0.5))
model.add(cv.Conv1D(filters=20,kernel_size=2,padding='same',activation='relu'))
model.add(pl.MaxPooling1D(pool_size=2))
model.add(Dropout(0.5))
model.add(cv.Conv1D(filters=30,kernel_size=2,padding='same',activation='relu'))
model.add(pl.MaxPooling1D(pool_size=2))
model.add(Dropout(0.5))
model.add(keras.layers.core.Flatten())
model.add(Dense(128, activation='relu'))
# model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
# model.add(Dropout(0.5))
model.add(BatchNormalization())
model.add(Dense(32, activation='relu'))

model.add(Dense(1, activation='linear'))
myAd = keras.optimizers.Adam(lr=0.0000001)
model.compile(loss='mae', optimizer='SGD', metrics=['mse'])

# trainning net
model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=20, batch_size=5,shuffle=True)
scores = model.evaluate(x_test, y_test, verbose=0)
print(scores)
#plot_model(model,to_file='d:/model.png')
