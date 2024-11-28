'''
Training on noisy data
'''
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
import os
AUTOTUNE = tf.data.AUTOTUNE

epochs = 100

## Dataset
n = 10000
noise_amplitude = .5
z_mean = np.random.uniform(0, 1., n)
z = z_mean + noise_amplitude*(np.random.rand(n)-.5)

dataset = tf.data.Dataset.from_tensor_slices((z, z_mean))
train_dataset, val_dataset = tf.keras.utils.split_dataset(dataset, right_size=0.2, shuffle=False)

## Prepare
train_dataset = train_dataset.shuffle(buffer_size=1000).batch(128).prefetch(buffer_size=tf.data.AUTOTUNE)
val_dataset = val_dataset.shuffle(buffer_size=1000).batch(128).prefetch(buffer_size=tf.data.AUTOTUNE)

## Model
model = Sequential([
    Dense(64, activation='leaky_relu', input_shape=(1,)),
    Dense(64, activation='leaky_relu'),
    Dense(1)
])
model.compile(optimizer="adam", loss='mean_squared_error', metrics=['mae'])

## Fit
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=epochs
)
#model.save(os.path.join(path,'z_'+P['filename_suffix']+'.tf'))

## Plot
plt.figure()
plt.plot(history.history['mae'])
plt.plot(history.history['val_mae'])
plt.title('Model MAE')
plt.ylabel('mae')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val', 'Val true'], loc='upper right')

## Test
z_mean = np.linspace(0, 1, n)
z = z_mean + noise_amplitude*(np.random.rand(n)-.5)
z_predict = model.predict(z)
plt.figure()
plt.plot([0,1], [0,1], 'k--')
plt.plot(z, z_predict, '.')
plt.show()
