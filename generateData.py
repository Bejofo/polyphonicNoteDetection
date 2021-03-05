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
    for _ in range(random.randint(1,2)):
	    filesToUse.append(random.choice(filenames))
    filesToUse = list(set(filesToUse))
    label = []
    for filename in filesToUse:
        noteName = filename_to_note(filename)
        noteName= librosa.note_to_midi(noteName)
        if noteName < 25 or noteName > 108:
          return generateExample(_)
        label.append(oneHotEncode(noteName))
    label = sum(label)
    if max(label) == 2:
        return generateExample(None)
    #wave = sum(sounds.raw[filename][:2048] for filename in filesToUse)
    freqs = abs(sum(sounds.cqt[filename][:,random.randint(0,6)] for filename in filesToUse))
    #features = np.concatenate((freqs,wave))
    return freqs,label


if __name__ =="__main__":
    i = int(input("Examples to generate?"))
    trainingData = np.zeros((i,84*5))
    labels =  np.zeros((i,128))
    pool = Pool()
    for d,l in pool.imap(generateExample,range(i)):
        trainingData[i-1] = np.array(d)
        labels[i-1] = np.array(l)
        i-=1
        # if i%200==0:
            # print(i)
    pool.close()
    pool.join()
    try:
        trainingData = np.vstack(np.load("trainingData.npy"),trainingData)
        labels =  np.vstack(np.load("labels.npy"),labels)
    except:
        pass
    np.save("trainingData.npy",trainingData)
    np.save("labels.npy",labels)
