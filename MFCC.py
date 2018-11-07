"""
Copyright by Dendi Suhubdy, 2018
"""
import librosa
from librosa.feature import mfcc

from Utils import fetch_sample_file


def extract_mfcc(audiofile):
    sample_rate, audio_data = fetch_sample_file(audiofile)
    mfccs = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=40)
    print(mfccs)
    return mfccs

if __name__ == "__main__":
    extract_mfcc("./results/LJ001-0001.wav")    
