'''
Train a network to estimate z from Paramecium image.

There was an issue with brightness_range: it can be done on int images (255), not normalized.

TODO:
- Metrics: on both z and z_mean
- Deal with images with no background subtraction

Ideas:
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
import pandas as pd
from time import time
import numpy as np
#tf.config.threading.set_intra_op_parallelism_threads(1)
#tf.config.threading.set_inter_op_parallelism_threads(1)
#tf.debugging.set_log_device_placement(True)
#tf.config.set_visible_devices([], 'GPU')
AUTOTUNE = tf.data.AUTOTUNE

root = tk.Tk()
root.withdraw()

### Folders
path = filedialog.askdirectory(initialdir=os.path.expanduser('~/Downloads/'), title='Choose a folder')
img_path = os.path.join(path, 'images')
label_path = os.path.join(path, 'labels.csv')

### Parameters
parameters = [('epochs', 'Epochs', 500),
              ('load_checkpoint', 'Load checkpoint', False),
              ('predict_true_z', 'Predict true z', False), # this exists only for synthetic datasets
              ('filename_suffix', 'Filename suffix', ''),
              ('background_subtracted', 'Background subtracted', True), # if it is background subtracted, the background is constant
              ('min_scaling', 'Minimum intensity scaling', 0.5),
              ('max_scaling', 'Maximum intensity scaling', 1.2)
              ]
param_dialog = (ParametersDialog(title='Enter parameters', parameters=parameters))
P = param_dialog.value
save_checkpoint = True
checkpoint_filename = os.path.join(path,'best_z_'+P['filename_suffix']+'.tf')
parameter_path = os.path.join(path, 'training_'+P['filename_suffix']+'.yaml')
history_path = os.path.join(path, 'history_'+P['filename_suffix']+'.csv')
batch_size = 64 # seems slightly faster than 32
validation_ratio = 0.2 # proportion of images used for validation

## Read data
df = pd.read_csv(label_path)

## Extract filenames and labels
filenames = df['filename'].values
mean_z = df['mean_z'].values.reshape(-1, 1)
second_metric = False
if 'z' in df:
    z = df['z'].values.reshape(-1, 1)
if P['predict_true_z']:
    labels = z # but it would be nice to have it as validation though
else:
    if False: #'z' in df: # For some reason, this doesn't work
        labels = np.hstack((mean_z, z))
        second_metric = True
    else:
        labels = mean_z
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
shape = image.shape
print("Image shape:", shape)

## Check whether background is white or black (assuming uint8)
if np.array(image).mean()<128: # black
    print('Black background')
    black_background = True
else:
    black_background = False

# Create a tf.data.Dataset from filenames and labels
dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
dataset = dataset.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)

## Make training and validation sets
train_dataset, val_dataset = tf.keras.utils.split_dataset(dataset, right_size=validation_ratio, shuffle=False)

## Data augmentation
class RandomIntensityScaling(tf.keras.layers.Layer):
    def __init__(self, min_scale=0.8, max_scale=1.2, **kwargs):
        super().__init__(**kwargs)
        self.min_scale = min_scale
        self.max_scale = max_scale

    def call(self, inputs, training=None):
        #if not training:  # Apply augmentation only during training
        #    return inputs

        scale = tf.random.uniform([], self.min_scale, self.max_scale)
        if black_background:
            scaled_image = inputs * scale
        else:
            scaled_image = 1. - (1.-inputs) * scale
        return tf.clip_by_value(scaled_image, 0.0, 1.0)

    def get_config(self):
        config = super().get_config()
        config.update({
            "min_scale": self.min_scale,
            "max_scale": self.max_scale,
        })
        return config

if P['background_subtracted']:
    scaling = RandomIntensityScaling(P['min_scaling'], P['max_scaling'])
else:
    scaling = layers.RandomBrightness(factor=[P['min_scaling'], P['max_scaling']], value_range=[0., 1.])
data_augmentation = tf.keras.Sequential([
    layers.Rescaling(1./255),
    #layers.RandomFlip("horizontal_and_vertical"), ### This crashes with the GPU!!
    scaling
    #layers.RandomRotation(1., fill_mode="constant", fill_value=1.-black_background*1.)
])
train_dataset = train_dataset.map(lambda x, y: (data_augmentation(x), y), num_parallel_calls=AUTOTUNE)
val_dataset = val_dataset.map(lambda x, y: (data_augmentation(x), y), num_parallel_calls=AUTOTUNE)

## Prepare
train_dataset = train_dataset.shuffle(buffer_size=1000).batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
val_dataset = val_dataset.shuffle(buffer_size=1000).batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
#train_dataset = train_dataset.batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
#val_dataset = val_dataset.batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)

## Metrics
def second_absolute_error(y_true, y_pred): # doesn't give the right result!
    return tf.reduce_mean(tf.abs(y_pred - y_true[:, 1]))

## Load weights and model from the checkpoint
if P['load_checkpoint']:
    print('Loading previous model')
    #model.load_weights(checkpoint_filename)
    model = tf.keras.models.load_model(checkpoint_filename, custom_objects={'second_absolute_error': second_absolute_error})
else:
    model = Sequential([
        Conv2D(32, (3, 3), activation='leaky_relu', input_shape=shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='leaky_relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='leaky_relu'),
        Dense(1)
    ])
    # model.summary()

## Compile the model
if second_metric:
    model.compile(optimizer="adam", loss='mean_squared_error', metrics=['mae', second_absolute_error])
else:
    model.compile(optimizer="adam", loss='mean_squared_error', metrics=['mae'])

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
    callbacks = [checkpoint]
else:
    callbacks = []

## Train
t1 = time()
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=P['epochs'],
    callbacks=callbacks
)
t2 = time()
P['time'] = t2-t1

## Evaluate
if second_metric:
    loss, mae, mae2 = model.evaluate(val_dataset)
    print(f'Validation loss: {loss}, Validation MAE (mean z): {mae}, Validation MAE (z): {mae2}')
else:
    loss, mae = model.evaluate(val_dataset)
    print(f'Validation loss: {loss}, Validation MAE: {mae}')
model.save(os.path.join(path,'z_'+P['filename_suffix']+'.tf'))

## Save training history
df = pd.DataFrame(history.history)
if P['load_checkpoint'] and os.path.exists(history_path):
    df.to_csv(history_path, mode='a', index=False, header=False) # add to existing history
else:
    df.to_csv(history_path)

## Plot
plt.plot(history.history['mae'])
plt.plot(history.history['val_mae'])
if second_metric:
    plt.plot(history.history['val_second_absolute_error'])
plt.title('Model MAE')
plt.ylabel('mae')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val', 'Val true'], loc='upper right')
plt.show()

## Save parameters
with open(parameter_path, 'w') as f:
    yaml.dump(P, f)
