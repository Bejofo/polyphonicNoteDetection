import librosa.display
import librosa
import matplotlib.pyplot as plt
import numpy as np 
import skimage.io
import codecs
import json

def generateToneFTT(midiNum):
	hertz = librosa.midi_to_hz(midiNum)
	sr = 22050
	tone = librosa.tone(hertz, sr=22050, length=22050)
	S = librosa.feature.melspectrogram(y=tone, sr=sr)
	return librosa.mel_frequencies(n_mels=len(S))

def oneHotEncode(n):
	ans = np.zeros(128)
	ans[n] = 1
	return ans 

trainingData = np.array([np.zeros(128)])
labels = np.array([np.zeros(128)])


for x in range(50,90):
	b=list(generateToneFTT(x))
	trainingData = np.concatenate((trainingData, [b]))
	labels = np.concatenate((labels, [oneHotEncode(x)]))
np.savetxt("trainingData.txt",trainingData)
np.savetxt("labels.txt",labels)