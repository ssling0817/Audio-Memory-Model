import librosa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from keras import layers, Input
from keras.models import Sequential, Model
from keras.layers import LSTM, Concatenate, Dense, Dropout
from keras import regularizers
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import pickle as pk


from tensorflow.python.keras.callbacks import EarlyStopping
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)
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
    #fourier_tempogram  = librosa.feature.fourier_tempogram(y=x, sr=sr)

    
    #l = [np.mean(chroma_cqt), np.mean(chroma_cens), np.mean(melspectrogram), np.mean(spectral_contrast), np.mean(spectral_flatness), np.mean(tonnetz), np.mean(chromagram),np.mean(rmse), np.mean(spectral_centroids), np.mean(spectral_bandwidth), np.mean(spectral_rolloff), np.mean(zcr), np.mean(spectral_bandwidth_3), np.mean(spectral_bandwidth_4), np.mean(tempo)]
    #l = [np.mean(chroma_cqt), np.mean(chroma_cens), np.mean(melspectrogram), np.mean(spectral_contrast), np.mean(spectral_flatness), np.mean(tonnetz), np.mean(chromagram),np.mean(rmse), np.mean(spectral_centroids), np.mean(spectral_bandwidth), np.mean(spectral_rolloff), np.mean(zcr), np.mean(tempo)]
    l = [np.mean(tempogram), np.mean(chroma_cqt), np.mean(chroma_cens), np.mean(melspectrogram), np.mean(spectral_contrast), np.mean(spectral_flatness), np.mean(tonnetz), np.mean(chromagram),np.mean(rmse), np.mean(spectral_centroids), np.mean(spectral_bandwidth), np.mean(spectral_rolloff), np.mean(zcr), np.mean(tempo)]
    

    for e in mfcc:
        l.extend([np.mean(e)]) 
    fea = np.array(l)

    return fea

def MemoryModel(feature, audio):

    feature_input = Input(shape=(feature.shape[1],))
    audio_input = Input(shape=(1, audio.shape[2]))
    
    feature_model_1 = Dense(512,kernel_regularizer=regularizers.l2(0.001))(feature_input)
    feature_model_2 = Dense(256, activation='relu',kernel_regularizer=regularizers.l2(0.001))(feature_model_1)
    feature_model_3 = Dense(128, activation='relu',kernel_regularizer=regularizers.l2(0.001))(feature_model_2)
    feature_model_4 = Dense(64, activation='relu')(feature_model_3)
    feature_model = Dense(32, activation='relu')(feature_model_4)

    audio_model_1 = LSTM(128, return_sequences=True)(audio_input)
    audio_model_3 = LSTM(64, return_sequences=True)(audio_model_1)
    audio_model = LSTM(32)(audio_model_3)

    final_model_1 = Concatenate(axis = 1)([feature_model, audio_model])
    final_model_2 = Dense(64, activation='relu')(final_model_1)
    final_model_drop1 = Dropout(0.4)(final_model_2)
    final_model_3 = Dense(32, activation='relu')(final_model_drop1)
    final_model_drop = Dropout(0.2)(final_model_3)
    final_model = Dense(1, activation='sigmoid')(final_model_drop)


    model = Model(inputs=[feature_input, audio_input], outputs=final_model)
    opt = keras.optimizers.RMSprop(learning_rate=0.0001)
    model.compile(optimizer=opt,loss='mse', metrics=['mean_squared_error'])
    return model

train_csv = pd.read_csv('train.csv')

details = pd.read_csv('user_details.csv')
gt = train_csv['score'] 
total_gt = np.array(gt)
track = train_csv['track']
total_fea = np.empty(shape=(0,34))
total_audio = np.empty(shape=(0,110250))

for filename in track:
    file_path = './audios/clips/' + filename
    # read audio to numpy arrays using librosa
    x, sr = librosa.load(file_path,  mono=True)
    fea = compute_feature(x, sr)    
    fea = fea.reshape((1, fea.shape[0]))
    aud = x.reshape((1, 110250))
    
    total_fea = np.concatenate((total_fea, fea))
    total_audio = np.concatenate((total_audio, aud))
    print('total_fea shape', total_fea.shape, ' total_audio shape', total_audio.shape)
   
print('total_gt shape', total_gt.shape)
total_audio = np.reshape(total_audio, (total_audio.shape[0], 1, total_audio.shape[1]))

fea_train, fea_test, y_train, y_test = train_test_split(total_fea, total_gt, test_size=0.2)
audio_train, audio_test, y_train, y_test = train_test_split(total_audio, total_gt, test_size=0.2)

model = MemoryModel(fea_train, audio_train)

checkpoint_filepath = 'weight.{epoch:02d}-{val_loss:.2f}.h5'
model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    monitor='val_loss',
    mode='min',
    save_best_only=True)

#history = model.fit(X_train,y_train,epochs=200, batch_size=1,validation_data=(X_test,y_test))#,callbacks= [early_stopping_monitor])
history = model.fit([fea_train, audio_train],y_train,epochs=200, batch_size=2,validation_data=([fea_test, audio_test],y_test),callbacks=[model_checkpoint_callback])#,callbacks= [early_stopping_monitor])

predictions = model.predict([fea_test, audio_test])
print('prediction', predictions)
print('gt', y_test)



"""with open('model.joblib', 'wb') as f:
    pk.dump(model, f)"""
