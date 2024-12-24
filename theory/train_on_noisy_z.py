'''
Training on noisy z, using mean z in data.
This shows boundary effects when estimating the expectation from the noisy value.
'''
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tkinter import filedialog
import tkinter as tk
from gui.gui import *
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
import os
import pandas as pd
import seaborn as sns
AUTOTUNE = tf.data.AUTOTUNE

epochs = 500

root = tk.Tk()
root.withdraw()

### Folders
path = filedialog.askdirectory(initialdir=os.path.expanduser('~/Downloads/'), message='Choose a dataset folder')
label_path = os.path.join(path, 'labels.csv')

## Read data
df = pd.read_csv(label_path)

## Dataset
noise_amplitude = 240. # 260 - cilia; however, shouldn't I consider a 1.33 factor? I think actually that would tend to reduce noise
z_mean = df['mean_z'].values
m, M = z_mean.min(), z_mean.max()
#z_mean = np.random.uniform(m, M, len(z_mean))
z = z_mean + noise_amplitude*(np.random.rand(len(z_mean))-.5)

dataset = tf.data.Dataset.from_tensor_slices((z, z_mean))
train_dataset, val_dataset = tf.keras.utils.split_dataset(dataset, right_size=0.2, shuffle=False)

## Prepare
train_dataset = train_dataset.shuffle(buffer_size=1000).batch(128).prefetch(buffer_size=tf.data.AUTOTUNE)
val_dataset = val_dataset.shuffle(buffer_size=1000).batch(128).prefetch(buffer_size=tf.data.AUTOTUNE)

## Model
model = Sequential([
    Dense(64, activation='leaky_relu', input_shape=(1,)),
    #Dense(64, activation='leaky_relu'),
    Dense(1)
])
model.compile(optimizer=Adam(learning_rate=1e-4), loss='mean_squared_error', metrics=['mae'])

## Fit
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=epochs
)
#model.save(os.path.join(path,'z_'+P['filename_suffix']+'.tf'))

## Plot
sns.histplot(df['mean_z'], kde=True)

plt.figure()
plt.plot(history.history['mae'])
plt.plot(history.history['val_mae'])
plt.title('Model MAE')
plt.ylabel('mae')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val', 'Val true'], loc='upper right')

## Test
n = 1000
print('Mean z:', m, 'to', M, 'um')
z_mean = np.linspace(m, M, n)
z = z_mean + noise_amplitude*(np.random.rand(n)-.5)
z_predict = model.predict(z)
plt.figure()
plt.plot([m-noise_amplitude, M+noise_amplitude], [m-noise_amplitude, M+noise_amplitude], 'k--')
plt.plot(z, z_predict, '.')
plt.xlabel('z')
plt.ylabel('estimate')
plt.show()
