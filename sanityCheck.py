import keras
import librosa
import math
from generateData import generateTrainingData

model = keras.models.load_model('firstmodel')

def unOneHotEncode(vec):
    ans = []
    for k,v in enumerate(vec):
        if math.isclose(v,1):
            ans.append(k)
    return ans

data,answerKey = generateTrainingData()
predictions = model.predict([data])
temp =sorted( zip(predictions,range(1,129)), key=lambda x:x[0])
print(unOneHotEncode)
for x in temp:
    print(x)
