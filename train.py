from keras.metrics import categorical_accuracy
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.layers import Conv1D

from sklearn.model_selection import train_test_split
import keras
import numpy as np 
import datetime


X = np.loadtxt("trainingData.txt")
y = np.loadtxt("labels.txt")

X_train,X_valid,y_train,y_valid = train_test_split(X,y,test_size=0.2,random_state=1)

model = Sequential()

model.add(Dense(256, input_dim=168))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Dense(128))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Dense(128))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Dense(128))
model.add(BatchNormalization())
model.add(Activation('sigmoid'))


opt = keras.optimizers.Adam(lr=0.0004)
model.compile(loss='binary_crossentropy', optimizer=opt,metrics=['accuracy'])
history = model.fit(
    X,
    y,
    batch_size=128,
    epochs=330,
    validation_data=(X_valid,y_valid)
    )
model.save("model5")

import matplotlib
import matplotlib.pyplot as plt
# Data for plotting
fig, ax = plt.subplots()
ax.plot(history.history['val_loss'])
ax.set(xlabel='epochs', ylabel='acc',
       title='Summary')
#ax.set_ylim([0,1])
ax.grid()
plt.show()
