from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
import keras
import numpy as np 

X = np.loadtxt("trainingData.txt")
y = np.loadtxt("labels.txt")

model = Sequential()
model.add(Dense(256, input_dim=128))
model.add(Activation('relu'))
model.add(Dense(128))
model.add(Activation('sigmoid'))

opt = keras.optimizers.Adam(lr=0.001)
model.compile(loss='binary_crossentropy', optimizer=opt,metrics=['accuracy'])
model.fit(X, y, batch_size=64, epochs=100)
model.save("secondmodel")
pred = model.predict(X)
np.savetxt("prediction.txt",pred,fmt="%1.2f")