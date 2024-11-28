'''
Tune a model.
'''
import pandas as pd
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
from tkinter import filedialog
import tkinter as tk
from gui.gui import *
import yaml
import pandas as pd
from time import time
import numpy as np
from scipy import stats
import keras_tuner as kt
AUTOTUNE = tf.data.AUTOTUNE

root = tk.Tk()
root.withdraw()

### Folders
path = filedialog.askdirectory(initialdir=os.path.expanduser('~/Downloads/'), title='Choose a folder')
img_path = os.path.join(path, 'images')
label_path = os.path.join(path, 'labels.csv')

### Parameters
parameters = [('epochs', 'Epochs', 10)
              ]
param_dialog = (ParametersDialog(title='Enter parameters', parameters=parameters))
P = param_dialog.value
parameter_path = os.path.join(path, 'tuning.yaml')
batch_size = 64
validation_ratio = 0.2 # proportion of images used for validation

## Read data
df = pd.read_csv(label_path)

## Extract filenames and labels
filenames = df['filename'].values
labels = df['z'].values
filenames = [os.path.join(img_path, name) for name in filenames]
n = len(filenames)

## Load images

# Create a mapping function for loading and preprocessing images
def load_image(filename, label):
    # Load the image from the file path
    img = tf.io.read_file(filename)
    img = tf.image.decode_png(img, channels=1)
    #img = img / 255.0  # Normalize pixel values to [0, 1]
    return img, label

## Get image shape
image, _ = load_image(filenames[0], None)
image = np.array(image)
shape = image.shape
print("Image shape:", shape)

# Create a tf.data.Dataset from filenames and labels
dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
dataset = dataset.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)

## Make training and validation sets
train_dataset, val_dataset = tf.keras.utils.split_dataset(dataset, right_size=validation_ratio, shuffle=False)

just_rescaling = layers.Rescaling(1./255)

train_dataset = train_dataset.map(lambda x, y: (just_rescaling(x), y), num_parallel_calls=AUTOTUNE)
val_dataset = val_dataset.map(lambda x, y: (just_rescaling(x), y), num_parallel_calls=AUTOTUNE)

## Prepare
train_dataset = train_dataset.shuffle(buffer_size=1000).batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
val_dataset = val_dataset.shuffle(buffer_size=1000).batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)

def build_model(hp):
    model = tf.keras.Sequential()
    model.add(layers.Input(shape=shape))

    # Tuning the number of filters in Conv2D layers
    for i in range(hp.Int('conv_layers', 1, 4)):  # Number of Conv2D layers
        model.add(layers.Conv2D(
            filters=hp.Int(f'filters_{i}', min_value=32, max_value=128),
            kernel_size=hp.Choice('kernel_size', values=[3, 5]),
            activation='leaky_relu',
            padding='same'
        ))
        model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(layers.Flatten())

    # Tuning the number of units in Dense layer
    model.add(layers.Dense(
        units=hp.Int('dense_units', min_value=32, max_value=256),
        activation='leaky_relu'
    ))

    #if hp.Boolean("dropout"):
    #    model.add(layers.Dropout(rate=0.25))

    model.add(layers.Dense(1))  # Output layer
    learning_rate = hp.Float("lr", min_value=1e-4, max_value=1e-2, sampling="log")

    # Compile the model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='mean_squared_error', metrics=['mae']
    )

    return model

tuner = kt.Hyperband(
    build_model,
    objective='val_mae',
    max_epochs=P['epochs'],
    hyperband_iterations=2,
    directory='hyperband',
    project_name='hyperband'
)

# Run the tuner
stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
tuner.search(train_dataset, validation_data=val_dataset, callbacks=[stop_early]) ## not sure of epochs=P['epochs']

# Get the best model
best_model = tuner.get_best_models(num_models=1)[0]
best_model.summary()

# Get the best hyperparameters
best_hyperparameters = tuner.get_best_hyperparameters(num_trials=1)[0]

print(f"Best hyperparameters: {best_hyperparameters.values}")

## Evaluate
loss, mae = best_model.evaluate(val_dataset)
print(f'Validation loss: {loss}, Validation MAE: {mae}')
best_model.save(os.path.join(path,'z_'+P['filename_suffix']+'.tf'))

## Save parameters
P.update(best_hyperparameters.values)
P['loss'] = loss
P['mae'] = mae
with open(parameter_path, 'w') as f:
    yaml.dump(P, f)
