import os

import cv2
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np


def padding(x: np.ndarray, limit: int) -> np.ndarray:
    if x.shape[1] > limit:
        out = x[:, :limit]
    elif x.shape[1] < limit:
        out = np.zeros((x.shape[0], limit))
        out[:, :x.shape[1]] = x
    else:
        out = x
    return out


def get_melspec(audio, sr=16000):
    y, sr = librosa.load(audio, sr=sr)
    X = librosa.stft(y)
    Xdb = librosa.amplitude_to_db(abs(X))
    # img = np.stack((Xdb,) * 3, -1)
    # img = img.astype(np.uint8)
    return Xdb


def get_mfccs(audio: str, limit=float('inf'), sr: int = 16000) -> np.ndarray:
    y, sr = librosa.load(audio, sr=sr)
    a = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    return a


def spectral_centroid(audio: str, sr: int = 16000):
    y, sr = librosa.load(audio, sr=sr)
    return librosa.feature.spectral_centroid(y=y, sr=sr)


def spectral_rolloff(audio: str, sr: int = 16000):
    y, sr = librosa.load(audio, sr=sr)
    return librosa.feature.spectral_rolloff(y=y, sr=sr)


def first_preproc(raw_path: str, out_path: str):
    X, Y = [], []
    for file in os.listdir(raw_path):
        source_file = os.path.join(raw_path, file)
        y, sr = librosa.load(source_file, sr=16000)
        x1 = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        x2 = librosa.feature.spectral_centroid(y=y, sr=sr)
        x3 = librosa.feature.spectral_rolloff(y=y, sr=sr)
        X.append(np.concatenate([x1, x2, x3]))
        Y.append(int(file.split('_')[0]))

    l = int(np.mean([len(x[0]) for x in X]))
    X = np.stack([padding(x, l) for x in X])
    Y = np.stack(Y)
    print(X.shape)
    print(Y.shape)
    os.makedirs(out_path,exist_ok=True)
    np.save(os.path.join(out_path, "input"), X)
    np.save(os.path.join(out_path, "label"), Y)


if __name__ == '__main__':
    db_path: str = os.path.abspath('audioDB')
    first_preproc(os.path.join(db_path, 'prepared'), os.path.join(db_path, 'preprocessing1'))
