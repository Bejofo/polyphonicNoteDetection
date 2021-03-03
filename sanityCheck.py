import librosa
import math
import numpy as np
import soundfile as sf
from multiprocessing import Pool
import os
import random
import cProfile
import functools
import librosa.display
import matplotlib.pyplot as plt


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


def unOneHotEncode(vec):
    ans = []
    for k,v in enumerate(vec):
        if math.isclose(v,1,rel_tol=1e-3):
            ans.append(k)
    return ans

def to_human(notes):
    return list(map(librosa.core.midi_to_note,notes))

def generateExample(_):
    filesToUse = []
    sr = 22050
    for _ in range(random.randint(1,3)):
	    filesToUse.append(random.choice(filenames))
    filesToUse = list(set(filesToUse))
    label = []
    noteNames = [] 
    for filename in filesToUse:
        noteName = filename_to_note(filename)
        noteNames.append(noteName)
        label.append(oneHotEncode(librosa.note_to_midi(noteName)))
    label = sum(label)
    if max(label) == 2:
        return generateExample(None)
    combinedWave = sum(sample_bank[filename] for filename in filesToUse)
    combinedWave/= len(filesToUse)
   
    # sf.write(f'{str(noteNames)}.wav', combinedWave, 22050, subtype='PCM_24')
    freqs = np.abs(librosa.cqt(
                combinedWave, 
                sr=sr, 
                fmin=librosa.note_to_hz('C1'),
                n_bins=84 * 2, bins_per_octave=12*2))

    plt.figure()
    librosa.display.specshow(librosa.amplitude_to_db(np.abs(freqs), ref=np.max))
    plt.show()

    return freqs,label


for _ in range(12):
    data,answerKey= generateExample(None)


