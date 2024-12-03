'''
Train a network to estimate z from Paramecium image.
Uses (deprecated) ImageGenerator instead of Keras dataset.

Lessons:
- Use as many examples as you can
- Adapt the learning rate when loss increases; but this doesn't always work great (stops too quickly)
- BatchNormalization works if learning is adaptively reduced

tensorflow-macos              2.13.0        update?
torch                         2.4.0
numpy                         1.24.3
python                        3.8.16        probably too old!
'''
import pandas as pd
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Rescaling, Dense, GlobalAveragePooling2D, BatchNormalization, LeakyReLU, Dropout, ReLU
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import matplotlib
from tkinter import filedialog
import tkinter as tk
from gui.gui import *
import yaml
import pandas as pd
from time import time
import numpy as np
from scipy import stats
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
import tqdm
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import sys
matplotlib.use('TkAgg')
AUTOTUNE = tf.data.AUTOTUNE

root = tk.Tk()
root.withdraw()

### Folders
path = filedialog.askdirectory(initialdir=os.path.expanduser('~/Downloads/'), message='Choose a dataset folder')
img_path = os.path.join(path, 'images')
label_path = os.path.join(path, 'labels.csv')

### Parameters
parameters = [('epochs', 'Epochs', 500),
              ('load_checkpoint', 'Load checkpoint', False),
              ('predict_sigma', 'Predict sigma', False), # this exists only for synthetic datasets
              ('filename_suffix', 'Filename suffix', '')
              ]
param_dialog = (ParametersDialog(title='Enter parameters', parameters=parameters))
P = param_dialog.value
save_checkpoint = True
checkpoint_filename = os.path.join(path,'best_z_'+P['filename_suffix']+'.tf')
parameter_path = os.path.join(path, 'training_'+P['filename_suffix']+'.yaml')
history_path = os.path.join(path, 'history_'+P['filename_suffix']+'.csv')
model_filename = os.path.join(path, 'model_'+P['filename_suffix']+'.txt')
batch_size = 32
validation_ratio = 0.2 # proportion of images used for validation

## Read data
df = pd.read_csv(label_path)

## Extract filenames and labels
filenames = df['filename'].values
if P['predict_sigma']:
    labels = df['sigma'].values
else:
    labels = df['mean_z'].values
filenames = [os.path.join(img_path, name) for name in filenames]
n = len(filenames)

## Load images

# Create a mapping function for loading and preprocessing images
def load_image(filename, black_background=True):
    # Load the image from the file path
    img = tf.io.read_file(filename)
    img = tf.image.decode_png(img, channels=1)
    img = tf.cast(img, dtype=tf.float32)
    if black_background:
        img = img / 255.0  # Normalize pixel values to [0, 1]
    else:
        img = 1.-img / 255.0  # Normalize pixel values to [0, 1]
    return img

## Get image shape
image = load_image(filenames[0])
image = np.array(image)
shape = image.shape
print("Image shape:", shape)

## Check whether background is white or black (assuming uint8)
most_frequent_value, _ = stats.mode(image.flatten())
print("Background value:", most_frequent_value)
if image.mean()<.5: # black
    print('Black background')
    black_background = True
else:
    black_background = False

## Create a dataset
images = np.array([load_image(os.path.join(img_path, file), black_background=black_background) for file in tqdm.tqdm(filenames)])
labels = np.array(labels)
print(len(images), len(labels))
print(images.min(), images.max(), np.std(images))


## Load weights and model from the checkpoint
if P['load_checkpoint']:
    print('Loading previous model')
    #model.load_weights(checkpoint_filename)
    model = tf.keras.models.load_model(checkpoint_filename)
else:
    # model = Sequential([ # tuned model, but I'm not sure, final receptive fields are too small
    #     Conv2D(75, (5, 5), activation='leaky_relu', input_shape=shape),
    #     MaxPooling2D((2, 2)),
    #     Conv2D(87, (5, 5), activation='leaky_relu'),
    #     MaxPooling2D((2, 2)),
    #     Flatten(),
    #     Dense(161, activation='leaky_relu'),
    #     Dense(1)
    # ])
    # model = Sequential([
    #     Conv2D(32, (3, 3), activation='relu', input_shape=shape), # , kernel_initializer='he_normal'
    #     MaxPooling2D((2, 2)),
    #     Conv2D(32, (3, 3), activation='relu'),
    #     MaxPooling2D((2, 2)),
    #     Conv2D(32, (3, 3), activation='relu'),
    #     MaxPooling2D((2, 2)),
    #     Conv2D(32, (3, 3), activation='relu'),
    #     MaxPooling2D((2, 2)),
    #     #GlobalAveragePooling2D(),
    #     Flatten(),
    #     #Dropout(0.5),
    #     Dense(128, activation='relu', kernel_initializer='he_normal'), # ou relu, à tester; aussi kernel_initializer=he_normal ou he_uniform
    #     Dense(1)
    # ])

    model = Sequential([
        Conv2D(32, (3, 3), kernel_initializer='he_normal', input_shape=shape, activation="relu"),
        #BatchNormalization(),
        MaxPooling2D((2, 2)),

        Conv2D(64, (3, 3), kernel_initializer='he_normal', activation="relu"),
        #BatchNormalization(),
        MaxPooling2D((2, 2)),

        # Conv2D(16, (3, 3), kernel_initializer='he_normal'),
        # # BatchNormalization(),
        # MaxPooling2D((2, 2)),
        #
        # Conv2D(16, (3, 3), kernel_initializer='he_normal'),
        # LeakyReLU(),
        # MaxPooling2D((2, 2)),
        #
        # Conv2D(16, (3, 3), kernel_initializer='he_normal'),
        # LeakyReLU(),
        # MaxPooling2D((2, 2)),

        #GlobalAveragePooling2D(),
        Flatten(),
        #Dropout(0.5),
        Dense(64, activation='relu', kernel_initializer='he_normal'),
        #Dense(128, activation='relu', kernel_initializer='he_normal'),
        #BatchNormalization(),
        Dense(1),
        #Rescaling(1000.) # useful?
    ])

model.summary()
with open(model_filename, "w") as f:
    sys.stdout = f  # Redirect output to the file
    model.summary()
    sys.stdout = sys.__stdout__  # Reset to default

## Compile the model
# model.compile(optimizer='rmsprop',
#               loss='mean_squared_error', metrics=['mae'])
model.compile(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.001), # default learning_rate .001
              loss='mean_squared_error', metrics=['mae'])

reduce_lr = ReduceLROnPlateau(monitor='val_loss',  # Monitor validation loss
                              factor=0.5,         # Reduce learning rate by half
                              patience=5, # this must be adapted to the dataset size
                              min_lr=1e-7,        # Minimum learning rate
                              verbose=1)

#early_stop = EarlyStopping(patience=10, verbose=1) # doesn't do what I thought...

## Define the ModelCheckpoint callback
checkpoint = ModelCheckpoint(
    checkpoint_filename,         # File path where the best weights will be saved
    monitor='val_loss',      # Metric to monitor for saving the best model
    save_best_only=True,      # Save only the best model weights
    mode='min',              # Mode for monitoring (min for loss, max for accuracy)
    verbose=0,                # Print information about the saving process
    save_weights_only=False
)

if save_checkpoint:
    callbacks = [checkpoint, reduce_lr]
else:
    callbacks = [reduce_lr]

## Train
t1 = time()
# Train the model
print("Starting fit")
history = model.fit(
    images, labels,
    batch_size=batch_size,
    validation_split=0.2,
    epochs=P['epochs'],
    callbacks=callbacks
)
t2 = time()
P['time'] = t2-t1

# Evaluate the model
# loss, mae = model.evaluate(val_generator)
# print(f'Validation loss: {loss}, Validation MAE: {mae}')
# P['loss'] = loss
# P['mae'] = mae
P['mae'] = history.history['mae'][-1]
P['val_mae'] = history.history['val_mae'][-1]
model.save(os.path.join(path,'z_'+P['filename_suffix']+'.tf'))

## Save training history
df = pd.DataFrame(history.history)
if P['load_checkpoint'] and os.path.exists(history_path):
    df.to_csv(history_path, mode='a', index=False, header=False) # add to existing history
else:
    df.to_csv(history_path, index=False)

## Plot
plt.plot(history.history['mae'])
plt.plot(history.history['val_mae'])
plt.title('Model MAE')
plt.ylabel('mae')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper right')
plt.show()

## Save parameters
with open(parameter_path, 'w') as f:
    yaml.dump(P, f)
