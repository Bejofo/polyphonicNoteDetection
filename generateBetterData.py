import os
import librosa
import random
import numpy as np
import pickle

sample_bank = {}
for filename in os.listdir("samples"):
    y,sr = librosa.load(f"samples/{filename}")
    sample_bank[filename] = y 

def oneHotEncode(n):
	ans = np.zeros(128)
	ans[n] = 1
	return np.array(ans) 

def generateExample():
    filesToUse = []
    sr = 22050
    for _ in random.randint(1,3):
	    filesToUse.append(random.choice(list(sample_bank.values())))
    filesToUse = list(set(filesToUse))
    label = []
    for filename in filesToUse:
        noteName = filename.split("-")[0]
        label.append(oneHotEncode(librosa.note_to_midi(noteName)))
    label = sum(label)
    combinedWave = sum(sample_bank[filename] for filename in filesToUse)
    combinedWave/= len(filesToUse)
    freqs = librosa.feature.melspectrogram(y=combinedWave, sr=sr)
    return freqs,label


trainingData = np.array([])
labels = np.array([])
i = int(input("Examples to generate?"))
for _ in range(i):
    d,l = generateExample()
    trainingData =np.concatenate((trainingData, [d]))
    labels = np.concatenate((labels,[l]))

np.savetxt("trainingData.txt",trainingData,fmt='%.5e')
np.savetxt("labels.txt",labels,fmt='%.1e')


    

