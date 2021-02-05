import librosa
import numpy as np 
import random
import time


def generateTrainingData():
	num_of_notes = random.randint(0,3)
	sample_rate = 22050
	waveform = np.zeros(sample_rate) # a second of audio
	label = np.zeros(128)
	notes = random.sample(range(1, 128), num_of_notes)
	for n in notes:
		hertz = librosa.midi_to_hz(n)
		waveform += librosa.tone(hertz, sr=sample_rate, length=sample_rate)
		label += oneHotEncode(n)		
	# sprinkle in some noise
	waveform += np.random.rand(sample_rate) * 10
	freqs = librosa.feature.melspectrogram(y=waveform, sr=sample_rate)
	return freqs,labels

def oneHotEncode(n):
	ans = np.zeros(128)
	ans[n] = 1
	return ans 

trainingData = np.array([np.zeros(128)])
labels = np.array([np.zeros(128)])


if __name__ == "__main__":
	SEED = int(time.time())
	random.seed(SEED)
	print(f"Using {SEED} as seed")
	for x in range(10):
		data,l=generateTrainingData()
		trainingData = np.concatenate((trainingData, [data]))
		labels = np.concatenate((labels, [l]))
	np.savetxt("trainingData.txt",trainingData,fmt='%.5e')
	np.savetxt("labels.txt",labels,fmt='%.1e')