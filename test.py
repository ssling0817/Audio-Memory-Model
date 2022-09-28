import librosa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.models import Sequential
from sklearn.ensemble import RandomForestRegressor
import pickle as pk
import csv
import argparse


def compute_feature(x, sr):
    #tempo
    tempo, beat_frames = librosa.beat.beat_track(y=x, sr=sr)
    #(1)
    zcr = librosa.feature.zero_crossing_rate(x)
    #(2) amplitude
    X = librosa.stft(x)
    Xdb = librosa.amplitude_to_db(abs(X))
    #(3) where the ” center of mass” for a sound is located
    spectral_centroids = librosa.feature.spectral_centroid(x, sr=sr)[0]
    #(4) signal shape
    spectral_rolloff = librosa.feature.spectral_rolloff(x+0.01, sr=sr)[0]
    #(5) the order-p spectral bandwidth
    spectral_bandwidth = librosa.feature.spectral_bandwidth(x+0.01, sr=sr)[0]
    spectral_bandwidth_3 = librosa.feature.spectral_bandwidth(x+0.01, sr=sr, p=3)[0]
    spectral_bandwidth_4 = librosa.feature.spectral_bandwidth(x+0.01, sr=sr, p=4)[0]

    #(6) the overall shape of a spectral envelope, shape (20, 216)
    mfcc = librosa.feature.mfcc(x, sr=sr)
    #(7) how much energy of each pitch class
    chromagram = librosa.feature.chroma_stft(x, sr=sr)#, hop_length=hop_length)
    #(8)
    rmse = librosa.feature.rms(y=x)
    #add
    chroma_cqt = librosa.feature.chroma_cqt(y=x, sr=sr)
    chroma_cens = librosa.feature.chroma_cens(y=x, sr=sr)
    melspectrogram = librosa.feature.melspectrogram(y=x, sr=sr)
    spectral_contrast = librosa.feature.spectral_contrast(y=x, sr=sr)
    spectral_flatness = librosa.feature.spectral_flatness(y=x)
    tonnetz = librosa.feature.tonnetz(y=x, sr=sr)
    tempogram = librosa.feature.tempogram(y=x, sr=sr)
    #l = [np.mean(chromagram),np.mean(rmse), np.mean(spectral_centroids), np.mean(spectral_bandwidth), np.mean(spectral_rolloff), np.mean(zcr)]
    #l = [np.mean(chromagram),np.mean(rmse), np.mean(spectral_centroids), np.mean(spectral_bandwidth), np.mean(spectral_rolloff), np.mean(zcr), np.mean(spectral_bandwidth_3), np.mean(spectral_bandwidth_4), np.mean(tempo), np.mean(beat_frames)]
    #l = [np.mean(chroma_cqt), np.mean(chroma_cens), np.mean(melspectrogram), np.mean(spectral_contrast), np.mean(spectral_flatness), np.mean(tonnetz), np.mean(chromagram),np.mean(rmse), np.mean(spectral_centroids), np.mean(spectral_bandwidth), np.mean(spectral_rolloff), np.mean(zcr), np.mean(spectral_bandwidth_3), np.mean(spectral_bandwidth_4), np.mean(tempo)]
    #l = [np.mean(chroma_cqt), np.mean(chroma_cens), np.mean(melspectrogram), np.mean(spectral_contrast), np.mean(spectral_flatness), np.mean(tonnetz), np.mean(chromagram),np.mean(rmse), np.mean(spectral_centroids), np.mean(spectral_bandwidth), np.mean(spectral_rolloff), np.mean(zcr), np.mean(tempo)]
    l = [np.mean(tempogram), np.mean(chroma_cqt), np.mean(chroma_cens), np.mean(melspectrogram), np.mean(spectral_contrast), np.mean(spectral_flatness), np.mean(tonnetz), np.mean(chromagram),np.mean(rmse), np.mean(spectral_centroids), np.mean(spectral_bandwidth), np.mean(spectral_rolloff), np.mean(zcr), np.mean(tempo)]
    
    for e in mfcc:
        l.extend([np.mean(e)]) 
    fea = np.array(l)

    return fea

parser = argparse.ArgumentParser()
parser.add_argument("--filename")
args = parser.parse_args()


#with open('model.joblib', 'rb') as f:
#    RFregressor = pk.load(f)
RFregressor = keras.models.load_model("best_1.h5")
total_fea = np.empty(shape=(0,34))
total_audio = np.empty(shape=(0,110250))
#test_csv = pd.read_csv('test.csv')
track = []
#track = test_csv["track"]

#for filename in track:
for file_path in glob.glob(args.filename+'/*.wav'):
    track.append(file_path)
    #file_path = './audios/clips/' + filename
    x, sr = librosa.load(file_path,  mono=True)
    fea = compute_feature(x, sr)    
    fea = fea.reshape((1, fea.shape[0]))
    aud = x.reshape((1, 110250))
    total_fea = np.concatenate((total_fea, fea))
    total_audio = np.concatenate((total_audio, aud))
total_audio = np.reshape(total_audio, (total_audio.shape[0], 1, total_audio.shape[1]))
print('total_fea shape', total_fea.shape, ' total_audio shape', total_audio.shape)
predictions = RFregressor.predict([total_fea, total_audio])

with open('file/output.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['track', 'score'])
    for i in range(predictions.shape[0]):
        writer.writerow([track[i], predictions[i][0]])

#print(RFregressor.summary())