import os
import librosa
sample_bank = {}
for filename in os.listdir("samples"):
    y,sr = librosa.load(f"samples/{filename}")
    sample_bank[filename] = y 
