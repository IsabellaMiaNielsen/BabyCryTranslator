from os.path import dirname, join as pjoin
import os
import matplotlib.pyplot as plt
import numpy as np
from playsound import playsound
import librosa
import json
import pickle


def plot_sound_wave(y, samplerate):
    librosa.display.waveshow(y, sr=samplerate, x_axis='s')
    length = y.shape[0] / samplerate
    print(f"length = {length}s")

def plot_spectrogram(spectrogram, title):
    fig, ax = plt.subplots(nrows=2, sharex=True)
    img = librosa.display.specshow(spectrogram, 
                                   x_axis='time', y_axis='mel', fmax=8000, ax=ax[0])
    fig.colorbar(img, ax=[ax[0]])
    ax[0].set(title=title)
    ax[0].label_outer()
    img = librosa.display.specshow(mfccs, x_axis='time', ax=ax[1])
    fig.colorbar(img, ax=[ax[1]])
    ax[1].set(title='MFCC')
    plt.show()

#Compare different DCT bases
def compare_DCT_bases(y, sr):
    m_slaney = librosa.feature.mfcc(y=y, sr=sr, dct_type=2)
    m_htk = librosa.feature.mfcc(y=y, sr=sr, dct_type=3)
    fig, ax = plt.subplots(nrows=2, sharex=True, sharey=True)
    img1 = librosa.display.specshow(m_slaney, x_axis='time', ax=ax[0])
    ax[0].set(title='RASTAMAT / Auditory toolbox (dct_type=2)')
    fig.colorbar(img1, ax=[ax[0]])
    img2 = librosa.display.specshow(m_htk, x_axis='time', ax=ax[1])
    ax[1].set(title='HTK-style (dct_type=3)')
    fig.colorbar(img2, ax=[ax[1]])
    plt.show()

def plot_rms(rms):
    fig, ax = plt.subplots(nrows=2, sharex=True)
    times = librosa.times_like(rms)
    ax[0].semilogy(times, rms[0], label='RMS Energy')
    ax[0].set(xticks=[])
    ax[0].legend()
    ax[0].label_outer()
    librosa.display.specshow(librosa.amplitude_to_db(S, ref=np.max),
                            y_axis='log', x_axis='time', ax=ax[1])
    ax[1].set(title='log Power spectrogram')
    plt.show()

def play_sound(wav_fname): 
    playsound(wav_fname)

data_dir = "donateacry-corpus-master/donateacry_corpus_cleaned_and_updated_data"
folders = ["belly_pain", "burping", "discomfort", "hungry", "tired"]
belly_pain = []
burping = []
discomfort = []
hungry = []
tired = []
labels = [belly_pain, burping, discomfort, hungry, tired]
data = {
    "belly_pain": {},
    "burping": {}, 
    "discomfort": {},
    "hungry": {}, 
    "tired": {}
}

for folder, label in zip(folders, labels):
    directory = pjoin(data_dir, folder)
    for filename in os.listdir(directory):
        wav_fname = pjoin(directory, filename)
        y, sr = librosa.load(wav_fname)

        # Extract pitches
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        # Extract Chroma features (Pitch class profiles)
        chroma_vector = librosa.feature.chroma_stft(y=y, sr=sr)
        # Extract zero-crossing rate
        zero_crossing_rate = librosa.feature.zero_crossing_rate(y)
        # Extract Mel-frequency cepstral coefficients
        mfccs = librosa.feature.mfcc(y=y, sr=sr)
        # Extract log-power mel spectorgram
        mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
        mel_spectrogram_db = librosa.power_to_db(mel_spectrogram)
        log_mel_spec = librosa.feature.mfcc(S=mel_spectrogram_db) 
        # Extract energy
        S = librosa.magphase(librosa.stft(y, window=np.ones, center=False))[0]
        rms_energy = librosa.feature.rms(S=S)

        #plot_rms(rms_energy)
        #plot_spectrogram(mel_spectrogram, 'Mel spectrogram')
        #plot_spectrogram(mel_spectrogram_db, 'Mel spectrogram db')
        #plot_spectrogram(log_mel_spec, 'Log Mel spectrogram')
        #compare_DCT_bases(y, sr)

        # Collect data
        
        data[folder][filename] = {
            "sound_array": y.tolist(),
            "samplerate": sr,
            "pitches": pitches.tolist(), 
            "magnitudes": magnitudes.tolist(),
            "chroma_features": chroma_vector.tolist(),
            "zero_crossing_rate": zero_crossing_rate.tolist(),
            "mfccs": mfccs.tolist(),
            #"mel_spectrogram": mel_spectrogram.tolist(),
            #"mel_spectrogram_db": mel_spectrogram_db.tolist(),
            "log_mel_spec": log_mel_spec.tolist(),
            "rms_energy": rms_energy.tolist()
        }
        
        #plot_sound_wave(y, samplerate)

        #play_sound(wav_fname)
        #stop = input("Stop?")
        #if stop == "y":
        #    break

with open('extracted_features.pkl', 'wb') as fp:
    pickle.dump(data, fp)
    print("dictionary saved successfully to file")
