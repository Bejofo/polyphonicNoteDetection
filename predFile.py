import librosa
import librosa.display
import matplotlib.pyplot as plt
import keras
import librosa
import math
import numpy as np
import mido
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


model = keras.models.load_model('model5')

def unOneHotEncode(vec):
    ans = []
    for k,v in enumerate(vec):
        if math.isclose(v,1,rel_tol=0.2):
            ans.append(k)
    return ans

def to_human(notes):
    return list(map(librosa.core.midi_to_note,notes))

waveform,sample_rate = librosa.load("emptyTown.wav")
waveform =  librosa.effects.harmonic(waveform)
freqs = np.abs(librosa.cqt(
                waveform, 
                sr=sample_rate, 
                fmin=librosa.note_to_hz('C1'),
                n_bins=84 * 5, bins_per_octave=12*5))
y = ""
predictions = model.predict(np.array(freqs).transpose())
deltaT = 0 
mid = mido.MidiFile()
track = mido.MidiTrack()
mid.tracks.append(track)
notesOn = []
for p in predictions:
    deltaT += 1
    predictedNotes = unOneHotEncode(p)
    # print(to_human(predictedNotes))
    for note in predictedNotes:
        if note in notesOn:
            continue
        notesOn.append(note)
        track.append(mido.Message('note_on', note=note, time=deltaT))
    for note in notesOn:
        if note in predictedNotes:
            continue
        notesOn.remove(note)
        track.append(mido.Message('note_off', note=note, time=deltaT))
mid.save('emptyTown.mid')
# f = open("transcription.txt",'w')
# f.write(y)
# f.close()
    
