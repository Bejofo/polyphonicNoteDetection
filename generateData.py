import librosa
import numpy as np 
import random
import time
import soundfile

#unused, and proablly wrong
def unOneHotEncode(vec):
    ans = []
    for k,v in enumerate(vec):
        if math.isclose(v,1):
            ans.append(k)
    return ans


def generateTrainingData():
	num_of_notes = random.randint(1,3)
	sample_rate = 22050
	waveform = np.zeros(sample_rate) # a second of audio
	label = np.zeros(128)
	human_readable_label = []
	notes = random.sample(range(30, 120), num_of_notes)
	for n in notes:
		hertz = librosa.midi_to_hz(n)
		waveform += librosa.tone(hertz, sr=sample_rate, length=sample_rate)
		label += oneHotEncode(n)		
		human_readable_label.append(librosa.midi_to_note(n))
	# sprinkle in some noise
	waveform += (np.random.rand(sample_rate)-0.5) * 0.01
	waveform/=num_of_notes # otherwise it clips
	freqs = librosa.feature.melspectrogram(y=waveform, sr=sample_rate)
	return freqs[:,1],label,human_readable_label,waveform

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
    print("How many samples to generate?")
    i = int(input())
    for x in range(i):
        data,l,human,waveform=generateTrainingData()
        trainingData = np.concatenate((trainingData, [data]))
        labels = np.concatenate((labels,[l]))
        if x % 100 == 0:
          #soundfile.write(f"{human}.wav", waveform, 22050, subtype='PCM_24')
          print(x)
    np.savetxt("trainingData.txt",trainingData,fmt='%.5e')
    np.savetxt("labels.txt",labels,fmt='%.1e')
