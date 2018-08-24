from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
import keras
import pandas as pd
from sklearn import preprocessing
from sklearn import model_selection



# prepare data
tb_path = r'E:\chenyw\毕业论文\湖南\changde.xls'
tb = pd.read_excel(tb_path, sheet_name='changde')
tb2 = tb
# tb2=tb[tb['dd']==5]
print(tb2.shape)
x = tb2.iloc[:, [0, 1, 2, 5]].values
y = tb2.iloc[:, -1].values
x = x.reshape(len(x), -1).astype('float32')
# y=y.reshape(len(y),-1).astype('float32')
t = keras.utils.to_categorical(y, num_classes=None)
x_train, x_test, y_train, y_test = model_selection.train_test_split(x, t, test_size=0.25)
print(tb2.head())
print(x[:2])


# make net
model = Sequential()
model.add(BatchNormalization(input_shape=(4,)))
# model.add(Conv1D(filters=2, kernel_size=2, strides=1,
#                  padding='same', activation='relu',
#                 ))
# model.add(MaxPooling1D(pool_size=2, padding='same'))
# model.add(Dropout(0.5))
# model.add(Conv1D(filters=20, kernel_size=2, strides=1, padding='same', activation='relu',
#                  kernel_constraint=keras.constraints.min_max_norm(min_value=0, max_value=1, rate=1.0)))
# model.add(MaxPooling1D(pool_size=2))
# # model.add(BatchNormalization(axis=1))
# model.add(Dropout(0.5))
# model.add(Conv1D(filters=50, kernel_size=2, strides=1, padding='same', activation='relu',
#                  kernel_constraint=keras.constraints.min_max_norm(min_value=0, max_value=1, rate=1.0)))
# model.add(MaxPooling1D(pool_size=2))
# model.add(Dropout(0.5))
# model.add(Flatten())
model.add(Dense(128, activation='relu'))
# model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
# model.add(Dropout(0.5))
model.add(BatchNormalization())
model.add(Dense(32, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(32, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(32,activation='relu'))
# model.add(Flatten())
model.add(Dense(6, activation='softmax'))
myAd = keras.optimizers.Adam(lr=0.0000001)
model.compile(loss='categorical_crossentropy', optimizer='SGD', metrics=['accuracy'])

# trainning net
model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=5, batch_size=30)
scores = model.evaluate(x_test, y_test, verbose=0)
print(scores)
