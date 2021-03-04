from multiprocessing import Pool
from SampleBank import SampleBank 
import os
import librosa
import random
import numpy as np
import cProfile
import functools

filenames = os.listdir("samples")
sounds = SampleBank()

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
    freqs = abs(sum(sounds.cqt[filename][:,1] for filename in filesToUse))
    return freqs,label


if __name__ =="__main__":
    trainingData = None
    labels =  None
    try:
        trainingData = np.load("trainingData.npy")
        labels =  np.load("labels.npy")
    except:
        pass
    i = int(input("Examples to generate?"))
    pool = Pool()
    for d,l in pool.imap(generateExample,range(i)):
        if (trainingData) is None or (labels is None):
            trainingData = [d]
            labels = [l]
        else:
            trainingData =np.stack((trainingData, d))
            labels = np.stack((labels,l))
        i-=1
        if i%100==0:
            print(i)
    pool.close()
    pool.join()
    np.save("trainingData.npy",trainingData)
    np.save("labels.npy",labels)


