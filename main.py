import librosa.display
import librosa
import matplotlib.pyplot as plt
import numpy as np 
import skimage.io
import codecs
import json

def generateToneFTT(midiNum):
	librosa.midi_to_hz(midiNum)
	sr = 22050
	tone = librosa.tone(440, sr=22050, length=22050)
	S = librosa.feature.melspectrogram(y=tone, sr=sr)
	return zip(librosa.mel_frequencies(n_mels=len(S)),S[:,1].tolist())

b=list(generateToneFTT(69))
json.dump(b, codecs.open("midi69.json", 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=True, indent=4)