import librosa
import librosa.display
import matplotlib.pyplot as plt
import keras
import librosa
import math
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


model = keras.models.load_model('secondmodel')

def unOneHotEncode(vec):
    ans = []
    for k,v in enumerate(vec):
        if math.isclose(v,1,rel_tol=1e-3):
            ans.append(k)
    return ans

def to_human(notes):
    return list(map(librosa.core.midi_to_note,notes))

waveform,sample_rate = librosa.load("test.wav")
freqs = librosa.feature.melspectrogram(y=waveform, sr=sample_rate)
y = ""
for i in range(0,freqs.shape[1],2):
    predictions = model.predict(np.array([freqs[:,i]]))
    x= unOneHotEncode(predictions[0])
    if i%100==0:
        print(f"{i}/{freqs.shape[1]}")
    y+= f"{to_human(x)}\n"
f = open("transcription.txt",'w')
f.write(y)
f.close()
    
