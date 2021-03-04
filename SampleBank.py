import librosa
import numpy as np
from multiprocessing import Pool
import os
import pickle
class SampleBank:
    PATH = "data.pickle"
    def __init__(self):
        if os.path.isfile(self.PATH):
            print("Sample Bank loaded")
            self.load()
            return
        self.raw = {}
        self.cqt = {}
        filenames = os.listdir("samples")
        pool = Pool()
        for f,data in pool.imap(load_into_bank,filenames):
            self.raw[f] = data
            self.cqt[f] = librosa.cqt(
                data, 
                fmin=librosa.note_to_hz('C1'),
                n_bins=84 * 2, bins_per_octave=12*2)
            print(f)
        pool.close()
        pool.join()
        self.save()
    
    def save(self):
        f = open(self.PATH,'wb')
        pickle.dump({
            "raw":self.raw,
            "cqt":self.cqt
        },f)
        f.close()
    
    def load(self):
        f = open(self.PATH,'rb')
        obj = pickle.load(f)
        f.close()
        self.raw = obj["raw"]
        self.cqt = obj["cqt"]


def load_into_bank(f):
    return f,librosa.load(f"samples/{f}",duration=1.5)[0]