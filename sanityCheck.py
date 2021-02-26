import keras
import librosa
import math
import numpy as np
from generateData import generateTrainingData

model = keras.models.load_model('secondmodel')

def unOneHotEncode(vec):
    ans = []
    for k,v in enumerate(vec):
        if math.isclose(v,1,rel_tol=1e-3):
            ans.append(k)
    return ans

def to_human(notes):
    return list(map(librosa.core.midi_to_note,notes))

for _ in range(20):
    data,answerKey,human,waveform = generateTrainingData()
    predictions = model.predict(np.array([data]))
    temp =sorted( zip(predictions,range(1,129)), key=lambda x:x[0])
    print(f"Expected:{to_human(unOneHotEncode(answerKey))}")
    print(f"Predicted:{to_human(unOneHotEncode(predictions[0]))}")
