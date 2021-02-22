from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from sklearn.model_selection import train_test_split
import keras
import numpy as np 

X = np.loadtxt("trainingData.txt")
y = np.loadtxt("labels.txt")

X_train,X_valid,y_train,y_valid = train_test_split(X,y,test_size=0.2,random_state=1)

model = Sequential()
model.add(Dense(256, input_dim=128))
model.add(Activation('relu'))
model.add(Dense(128))
model.add(Activation('sigmoid'))

opt = keras.optimizers.Adam(lr=0.001)
model.compile(loss='binary_crossentropy', optimizer=opt,metrics=['accuracy'])
model.fit(
    X,
    y,
    batch_size=64,
    epochs=100,
    validation_data=(X_valid,y_valid)
    )
model.save("secondmodel")
pred = model.predict(X)
np.savetxt("prediction.txt",pred,fmt="%1.2f")
