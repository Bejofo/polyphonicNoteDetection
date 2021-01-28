import librosa.display
import librosa
import matplotlib.pyplot as plt
import numpy as np 
import skimage.io
import codecs
import json

"""
def scale_minmax(X, min=0.0, max=1.0):
    X_std = (X - X.min()) / (X.max() - X.min())
    X_scaled = X_std * (max - min) + min
    return X_scaled


def spectrogram_image(y, sr, out, hop_length, n_mels):
    # use log-melspectrogram
    mels = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels,
                                            n_fft=hop_length*2, hop_length=hop_length)
    mels = numpy.log(mels + 1e-9) # add small number to avoid log(0)

    # min-max scale to fit inside 8-bit range
    img = scale_minmax(mels, 0, 255).astype(numpy.uint8)
    img = numpy.flip(img, axis=0) # put low frequencies at the bottom in image
    img = 255-img # invert. make black==more energy

    # save as PNG
    skimage.io.imsave(out, img)


if __name__ == '__main__':
    # settings
    hop_length = 512 # number of samples per time-step in spectrogram
    n_mels = 256 # number of bins in spectrogram. Height of image
    time_steps = 512 # number of time-steps. Width of image

    # load audio. Using example from librosa
    path = librosa.util.example_audio_file()
    y, sr = librosa.load("waltz.wav")
    out = 'out1.png'

    # extract a fixed length window
    start_sample = 0 # starting at beginning
    length_samples = time_steps*hop_length
    window = y[start_sample:start_sample+length_samples]
    
    # convert to PNG
    spectrogram_image(window, sr=sr, out=out, hop_length=hop_length, n_mels=n_mels)
    print('wrote file', out)
 """
 
 """
with open("ftt.txt","w") as f:
    for y_block in stream:
        nft = 2048*2
        D_block = librosa.stft(y_block,n_fft=nft, center=False)
        for i,n in enumerate(np.abs(D_block)):
            freqs = np.arange(0, 1 + nft / 2) * sr / nft
            if freqs[i] > 8000:
                break
            print(f"{freqs[i]} {n[1]}")
            #f.write(str(n[1]) + ",")
            b = np.abs(D_block[:,1]).tolist() # nested lists with same data, indices
            file_path = "./data.json" ## your path variable
            json.dump(b, codecs.open(file_path, 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=True, indent=4) ### this 
"""
 
sr = librosa.get_samplerate("sine.wav")
stream = librosa.stream("sine.wav",

                      block_length=1024,

                      frame_length=4096*2,

                      hop_length=1024)


def generateToneFTT(midiNum):
	librosa.midi_to_hz(midiNum)
	sr = 22050
	tone = librosa.tone(440, sr=22050, length=22050)
	S = librosa.feature.melspectrogram(y=tone, sr=sr)
	return zip(librosa.mel_frequencies(n_mels=len(S)),S[:,1].tolist())



b=list(generateToneFTT(69))
json.dump(b, codecs.open("midi69.json", 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=True, indent=4)