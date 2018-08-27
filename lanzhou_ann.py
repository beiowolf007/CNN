from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
import keras
import pandas as pd
from sklearn import preprocessing
from sklearn import model_selection

# prepare data
tb_path = r'E:\chenyw\毕业论文\甘肃\lanzhou.xls'
tb = pd.read_excel(tb_path, sheet_name='lanzhou')
tb2 = tb
# tb2=tb[tb['dd']==5]
print(tb2.shape)
x = tb2.iloc[:, 0:9].values
y = tb2.iloc[:, 9].values
# x = x.reshape(len(x), -1).astype('float32')
# y=y.reshape(len(y),-1).astype('float32')
x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.25)
print(tb2.head())
print(x[:2])
print(y[:3])

# make net
model = Sequential()
model.add(BatchNormalization(input_shape=(9,)))

model.add(Dense(128, activation='relu'))
# model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
# model.add(Dropout(0.5))
model.add(BatchNormalization())
model.add(Dense(32, activation='relu'))

model.add(Dense(1, activation='linear'))
myAd = keras.optimizers.Adam(lr=0.0000001)
model.compile(loss='mse', optimizer='Adam', metrics=['mse', 'acc'])

# trainning net
model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=40, batch_size=10)
scores = model.evaluate(x_test, y_test, verbose=0)
print(scores)
