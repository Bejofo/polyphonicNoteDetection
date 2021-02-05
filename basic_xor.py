from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD
import numpy as np 

X = np.loadtxt("traingData.txt")
y = np.loadtxt("labels.txt")

model = Sequential()
model.add(Dense(128, input_dim=128))
model.add(Activation('tanh'))
model.add(Dense(128))
model.add(Activation('sigmoid'))


sgd = SGD(lr=0.1)
model.compile(loss='binary_crossentropy', optimizer=sgd)

model.fit(X, y, show_accuracy=True, batch_size=1, nb_epoch=1000)
print(model.predict_proba(X))