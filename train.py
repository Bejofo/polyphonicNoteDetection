# from keras.metrics import categorical_accuracy
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from sklearn.model_selection import train_test_split
import keras
import numpy as np 
import datetime
print("hi")
X = np.load("trainingData.npy")
y = np.load("labels.npy")

X_train,X_valid,y_train,y_valid = train_test_split(X,y,test_size=0.2,random_state=1)

model = Sequential()

model.add(Dense(512, input_dim=84*5))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Dense(256))
model.add(BatchNormalization())
model.add(Activation('sigmoid'))


model.add(Dense(128))
model.add(BatchNormalization())
model.add(Activation('sigmoid'))

model.add(Dense(128))
model.add(BatchNormalization())
model.add(Activation('sigmoid'))


opt = keras.optimizers.Adam(lr=0.0008)
model.compile(loss='binary_crossentropy', optimizer=opt,metrics=['accuracy'])
print(model.summary())
callback = keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=3,verbose=1)
history = model.fit(
    X,
    y,
    batch_size=1024,
    epochs=100,
    validation_data=(X_valid,y_valid),
    callbacks=[callback]
    )
model.save("model5")

