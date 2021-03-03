from multiprocessing import Pool
import os
import librosa
import random
import numpy as np
import cProfile
import functools

sample_bank = {}
filenames = os.listdir("samples")
def load_into_bank(f):
    return f,librosa.load(f"samples/{f}",duration=1.5)[0]
pool = Pool()
for f,data in pool.imap(load_into_bank,filenames):
    sample_bank[f] = data
    print(f)
pool.close()
pool.join()

@functools.lru_cache(maxsize=None)
def oneHotEncode(n):
	ans = np.zeros(128)
	ans[n] = 1
	return np.array(ans) 


@functools.lru_cache(maxsize=None)
def filename_to_note(f):
	return f.split("-")[0]

def generateExample(_):
    filesToUse = []
    sr = 22050
    for _ in range(random.randint(1,3)):
	    filesToUse.append(random.choice(filenames))
    filesToUse = list(set(filesToUse))
    label = []
    for filename in filesToUse:
        noteName = filename_to_note(filename)
        label.append(oneHotEncode(librosa.note_to_midi(noteName)))
    label = sum(label)
    if max(label) == 2:
        return generateExample(None)
    combinedWave = sum(sample_bank[filename] for filename in filesToUse)
    combinedWave/= len(filesToUse)
    freqs = np.abs(librosa.cqt(
                combinedWave, 
                sr=sr, 
                fmin=librosa.note_to_hz('C1'),
                n_bins=84 * 2, bins_per_octave=12*2)[:,random.randint(0,6)])
    return freqs,label



if __name__ =="__main__":
    trainingData = np.loadtxt("trainingData.txt")
    labels =  np.loadtxt("labels.txt")
    i = int(input("Examples to generate?"))
    pool = Pool()
    for d,l in pool.imap(generateExample,range(i)):
        trainingData =np.concatenate((trainingData, [d]))
        labels = np.concatenate((labels,[l]))
    pool.close()
    pool.join()

    np.savetxt("trainingData.txt",trainingData,fmt='%.5e')
    np.savetxt("labels.txt",labels,fmt='%i')


