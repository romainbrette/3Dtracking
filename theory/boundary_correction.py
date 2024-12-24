'''
Correct boundary effects
'''
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
import os
AUTOTUNE = tf.data.AUTOTUNE

epochs = 50

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
    #Dense(64, activation='sigmoid'),
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

## Fit inverse function
z = np.linspace(-.5*noise_amplitude, 1+.5*noise_amplitude, n)
z_predict = model.predict(z)

## Model
inv_model = Sequential([
    Dense(64, activation='leaky_relu', input_shape=(1,)),
    #Dense(64, activation='sigmoid'),
    Dense(1)
])
inv_model.compile(optimizer=Adam(learning_rate=1e-4), loss='mean_squared_error', metrics=['mae'])

dataset = tf.data.Dataset.from_tensor_slices((z_predict, z))
train_dataset, val_dataset = tf.keras.utils.split_dataset(dataset, right_size=0.2, shuffle=False)

## Prepare
train_dataset = train_dataset.shuffle(buffer_size=1000).batch(128).prefetch(buffer_size=tf.data.AUTOTUNE)
val_dataset = val_dataset.shuffle(buffer_size=1000).batch(128).prefetch(buffer_size=tf.data.AUTOTUNE)

## Fit
history = inv_model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=epochs
)
#model.save(os.path.join(path,'z_'+P['filename_suffix']+'.tf'))

## Test
z = np.linspace(-.5*noise_amplitude, 1+.5*noise_amplitude, n)
z_predict = inv_model.predict(z)

plt.figure()
plt.plot([0,1], [0,1], 'k--')
plt.plot(z, z_predict, '.')
plt.xlabel('z')
plt.ylabel('estimate')
plt.show()
