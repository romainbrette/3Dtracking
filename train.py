'''
Train a network to estimate z from Paramecium image.

There was an issue with brightness_range: it can be done on int images (255), not normalized.

Roughly 45 s / epoch with the first model; the tuned model is horribly slow (400 s).

It seems that changing the network structure has little effect.

- maybe stratified dataset with eccentricity?
- MobileNetV3 or other deep net?
- Use preprocessed features as inputs, eg variance or Brenner gradient
    this is the mean squared 2-pixel wide horizontal difference (not rotation invariant!)
    note that this can be obtained with a filter of kernel size 3
- Not clear whether I should use adam or rmsprop


    Deprecated: `tf.keras.preprocessing.image.ImageDataGenerator` is not
    recommended for new code. Prefer loading images with
    `tf.keras.utils.image_dataset_from_directory` and transforming the output
    `tf.data.Dataset` with preprocessing layers. For more information, see the
    tutorials for [loading images](
    https://www.tensorflow.org/tutorials/load_data/images) and
    [augmenting images](
    https://www.tensorflow.org/tutorials/images/data_augmentation), as well as
    the [preprocessing layer guide](
    https://www.tensorflow.org/guide/keras/preprocessing_layers).

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
AUTOTUNE = tf.data.AUTOTUNE

root = tk.Tk()
root.withdraw()

### Folders
path = filedialog.askdirectory(initialdir=os.path.expanduser('~/Downloads/'), title='Choose a folder')
img_path = os.path.join(path, 'images')
label_path = os.path.join(path, 'labels.csv')
parameter_path = os.path.join(path, 'dataset', 'labels.yaml')

### Parameters
parameters = [('epochs', 'Epochs', 500),
              ('load_checkpoint', 'Load checkpoint', True), # this should be rendered as a tick
              ('predict_true_z', 'Predict true z', False), # this exists only for synthetic datasets
              ('filename_suffix', 'Filename suffix', '')
              ]
param_dialog = (ParametersDialog(title='Enter parameters', parameters=parameters))
P = param_dialog.value
save_checkpoint = True
checkpoint_filename = os.path.join(path,'best_z_'+P['filename_suffix']+'.tf')
batch_size = 64 # seems slightly faster than 32
validation_ratio = 0.2 # proportion of images used for validation

## Read data
df = pd.read_csv(label_path)

## Extract filenames and labels
filenames = df['filename'].values
if P['predict_true_z']:
    labels = df['z'].values # but it would be nice to have it as validation though
else:
    labels = df['mean_z'].values
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

image, _ = load_image(filenames[0], None)
shape = image.shape
print("Image shape:", shape)

# Create a tf.data.Dataset from filenames and labels
dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
dataset = dataset.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
#dataset = dataset.shuffle(buffer_size=1000).batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)

## Make training and validation sets
train_dataset, val_dataset = tf.keras.utils.split_dataset(dataset, right_size=validation_ratio, shuffle=False)

## Data augmentation
data_augmentation = tf.keras.Sequential([
    layers.Rescaling(1./255),
    layers.RandomFlip("horizontal_and_vertical")
    #layers.RandomBrightness(factor=[0.5, 2.], value_range=[0., 1.])  # not sure
    #layers.RandomRotation(0.2),
])
train_dataset = train_dataset.map(lambda x, y: (data_augmentation(x), y), num_parallel_calls=AUTOTUNE)
val_dataset = val_dataset.map(lambda x, y: (data_augmentation(x), y), num_parallel_calls=AUTOTUNE)

## Prepare
train_dataset = train_dataset.shuffle(buffer_size=1000).batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
val_dataset = val_dataset.shuffle(buffer_size=1000).batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)

model = Sequential([
    #Rescaling(1./255),
    Conv2D(32, (3, 3), activation='leaky_relu', input_shape=shape), # was 32; (3,3)
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='leaky_relu'), # was 64, (3, 3)
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='leaky_relu'), # was 64
    Dense(1)
])
model.summary()

# Load weights from the checkpoint
if P['load_checkpoint']:
    model.load_weights(checkpoint_filename)

# Compile the model
model.compile(optimizer="adam", loss='mean_squared_error', metrics=['mae']) # the tuner finds rmsprop rather than Adam

# Define the ModelCheckpoint callback
checkpoint = ModelCheckpoint(
    checkpoint_filename,         # File path where the best weights will be saved
    monitor='val_loss',      # Metric to monitor for saving the best model
    save_best_only=True,      # Save only the best model weights
    mode='min',              # Mode for monitoring (min for loss, max for accuracy)
    verbose=0,                # Print information about the saving process
    save_weights_only=False
)

if save_checkpoint:
    callbacks = [checkpoint]
else:
    callbacks = []

# Train the model
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=P['epochs'],
    callbacks=callbacks
)

# Evaluate the model
loss, mae = model.evaluate(val_dataset)
print(f'Validation loss: {loss}, Validation MAE: {mae}')

model.save(os.path.join(path,'z_'+P['filename_suffix']+'.tf'))

# Plot
plt.plot(history.history['mae'])
plt.plot(history.history['val_mae'])
plt.title('Model MAE')
plt.ylabel('mae')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper right')
plt.show()
