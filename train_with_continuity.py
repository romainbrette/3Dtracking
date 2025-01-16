'''
Trains a network to estimate z from Paramecium image.
Uses an additional loss of continuity of z (error in dz).

Not sure this is good given that the "true" z is in fact x, not z. Otherwise, normalize with variance.
And/or, have it on freely swimming cells (not straightforward to combine the two losses).
'''
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint
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
import sys
from tensorflow.keras.callbacks import ReduceLROnPlateau
from augmentation.augmentation import *
from models import *

AUTOTUNE = tf.data.AUTOTUNE

root = tk.Tk()
root.withdraw()

### Folders
path = filedialog.askdirectory(initialdir=os.path.expanduser('~/Downloads/'), message='Choose a dataset folder')
img_path = os.path.join(path, 'images')
label_path = os.path.join(path, 'labels.csv')

### Parameters
parameters = [('epochs', 'Epochs', 500),
              ('model', 'Model', list(models.keys())),
              ('load_checkpoint', 'Load checkpoint', False),
              ('filename_suffix', 'Filename suffix', ''),
              ('background_subtracted', 'Background subtracted', True), # if it is background subtracted, the background is constant # could be done automatically
              ('max_threshold', 'Maximum threshold', 0),
              ('min_scaling', 'Minimum intensity scaling', 1.),
              ('max_scaling', 'Maximum intensity scaling', 1.),
              ('continuity_weight', 'Continuity weight', 10.)
              ]
param_dialog = (ParametersDialog(title='Enter parameters', parameters=parameters))
P = param_dialog.value
continuity_weight = P['continuity_weight']
save_checkpoint = True
# Suffix
model_name = P['model']
suffix = model_name
if len(P['filename_suffix'])>0:
    suffix += '_'+P['filename_suffix']
checkpoint_filename = os.path.join(path,'best_z_'+suffix+'.tf')
parameter_path = os.path.join(path, 'training_'+suffix+'.yaml')
history_path = os.path.join(path, 'history_'+suffix+'.csv')
model_filename = os.path.join(path, 'model_'+suffix+'.txt')
dataset_parameter_path = os.path.join(path, 'labels.yaml')
batch_size = 128
validation_ratio = 0.2 # proportion of images used for validation

## Read data
df = pd.read_csv(label_path)

## Shuffle
#df = df.sample(frac=1).reset_index(drop=True)

## Read dataset parameters
with open(dataset_parameter_path, 'r') as f:
    P_dataset = yaml.safe_load(f)
# Normalization factor ignored for now
#normalization = P_dataset.get('normalization', 1.)
normalization = 1.
min_threshold = 0.
max_threshold = P['max_threshold']*normalization

## Extract filenames and labels
filenames = [os.path.join(img_path, filename) for filename in df['filename'].values]
mask = df['mask'].values
try:
    labels = df['mean_z'].values
except KeyError:
    labels = df['z'].values
n = len(filenames)
print('s.d. of z:', np.std(labels))
print('mean_abs_difference_metric:', np.mean((np.abs(labels[1:] - labels[:-1])* mask[:-1])))

for filename in filenames:
    if os.path.getsize(filename) == 0:
        print(filename, "is empty")

## Load images

# Create a mapping function for loading and preprocessing images
def load_image(filename):
    # Load the image from the file path
    img = tf.io.read_file(filename)
    img = tf.image.decode_png(img, channels=1)
    img = tf.cast(img, dtype=tf.float32)
    mean_intensity = tf.reduce_mean(img)
    mean_intensity = tf.maximum(mean_intensity, 1e-8)
    img = img/mean_intensity
    return img

def map_function(image_path, label, mask):
    image = load_image(image_path)
    return image, (label, mask)


## Get image shape
first_image = load_image(filenames[0])
first_image = np.array(first_image)
shape = first_image.shape
print("Image shape:", shape)

# Create a tf.data.Dataset from filenames and labels
dataset = tf.data.Dataset.from_tensor_slices((filenames, labels, mask))

dataset = dataset.map(lambda filename, label, mask: map_function(filename, label, mask),
                      num_parallel_calls=tf.data.AUTOTUNE)

## Make training and validation sets
train_dataset, val_dataset = tf.keras.utils.split_dataset(dataset, right_size=validation_ratio, shuffle=False)

## Data augmentation

if P['background_subtracted']:
    intensity_scaling = RandomIntensityScaling(P['min_scaling'], P['max_scaling'])
else:
    intensity_scaling = layers.RandomBrightness(factor=[P['min_scaling'], P['max_scaling']], value_range=[0., 1.])
if P['max_threshold']>0:
    data_augmentation = tf.keras.Sequential([
        #layers.RandomFlip("horizontal_and_vertical"), ### This crashes with the GPU!!
        RandomThreshold(min_threshold, max_threshold),
        intensity_scaling
        #layers.RandomRotation(1., fill_mode="constant", fill_value=1.-black_background*1.)  ### This crashes with the GPU!!
    ])
else:
    data_augmentation = tf.keras.Sequential([
        # layers.RandomFlip("horizontal_and_vertical"), ### This crashes with the GPU!!
        intensity_scaling
        # layers.RandomRotation(1., fill_mode="constant", fill_value=1.-black_background*1.)  ### This crashes with the GPU!!
    ])
#just_rescaling = layers.Rescaling(1./255)

train_dataset = train_dataset.map(lambda x, y: (data_augmentation(x), y), num_parallel_calls=AUTOTUNE)
#val_dataset = val_dataset.map(lambda x, y: (just_rescaling(x), y), num_parallel_calls=AUTOTUNE)
val_dataset = val_dataset.map(lambda x, y: (data_augmentation(x), y), num_parallel_calls=AUTOTUNE)

## Prepare
train_dataset = train_dataset.batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
val_dataset = val_dataset.batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)

def combined_loss(y_true, y_pred):
    """
    Custom loss that combines:
    1. The mean squared error for each output.
    2. The mean squared difference between the two outputs.

    Args:
    y_true: Tensor of shape (batch_size, 2), containing the true labels.
    y_pred: Tensor of shape (batch_size, 2), containing the predicted outputs.

    Returns:
    A scalar tensor representing the combined loss.
    """
    y_true, mask = tf.split(y_true, num_or_size_splits=2, axis=-1)

    # Compute the mean squared error for each output
    mse = tf.reduce_mean(tf.square(y_true - y_pred))

    # Compute the squared error for dz
    #squared_difference = tf.reduce_mean(tf.square(y_pred[1:] - y_pred[:-1]))
    #squared_difference = tf.reduce_mean(tf.multiply(tf.square(y_pred[1:] - y_pred[:-1]), mask[:-1]))
    squared_difference = tf.reduce_mean(tf.multiply(tf.square(y_pred[1:] - y_pred[:-1] - y_true[1:]+y_true[:-1]), mask[:-1]))

    # Combine the losses
    return mse + continuity_weight*squared_difference


def mean_abs_difference_metric(y_true, y_pred):
    """
    Metric to compute the absolute difference between successive z.

    Args:
    y_true: Tensor of shape (batch_size, 2), containing the true labels.
    y_pred: Tensor of shape (batch_size, 2), containing the predicted outputs.

    Returns:
    A scalar tensor representing the Mean Squared Difference.
    """
    _, mask = tf.split(y_true, num_or_size_splits=2, axis=-1)
    # Compute the absolute difference between successive z
    abs_difference = tf.reduce_mean(tf.multiply(tf.abs(y_pred[1:] - y_pred[:-1]), mask[:-1]))

    # Compute the mean
    return tf.reduce_mean(abs_difference)

def modified_mae(y_true, y_pred):
    y_true, _ = tf.split(y_true, num_or_size_splits=2, axis=-1)

    # Compute the mean
    return tf.reduce_mean(tf.abs(y_true - y_pred))

## Load weights and model from the checkpoint
if P['load_checkpoint']:
    print('Loading previous model')
    #model.load_weights(checkpoint_filename)
    model = tf.keras.models.load_model(checkpoint_filename, custom_objects={'modified_mae': modified_mae,
                                                                            'mean_abs_difference_metric': mean_abs_difference_metric,
                                                                            'combined_loss': combined_loss})
else:
    model = models[model_name](shape)

model.summary()
with open(model_filename, "w") as f:
    sys.stdout = f  # Redirect output to the file
    model.summary()
    sys.stdout = sys.__stdout__  # Reset to default

## Compile the model
model.compile(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.0001), #'adam', # default learning_rate .001
              loss=combined_loss, metrics=[modified_mae, mean_abs_difference_metric])

reduce_lr = ReduceLROnPlateau(monitor='loss',
                              factor=0.5,         # Reduce learning rate by half
                              patience=4, # this must be adapted to the dataset size
                              min_lr=1e-7,        # Minimum learning rate
                              verbose=1)

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
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=P['epochs'],
    callbacks=callbacks
)
t2 = time()
P['time'] = t2-t1

## Evaluate
loss, modified_mae, continuity_metric = model.evaluate(val_dataset)
print(f'Validation loss: {loss}, Validation MAE: {modified_mae}')
P['loss'] = loss
P['modified_mae'] = modified_mae
model.save(os.path.join(path,'z_'+suffix+'.tf'))

## Save training history
df = pd.DataFrame(history.history)
if P['load_checkpoint'] and os.path.exists(history_path):
    df.to_csv(history_path, mode='a', index=False, header=False) # add to existing history
else:
    df.to_csv(history_path)

## Plot
plt.plot(history.history['modified_mae'])
plt.plot(history.history['val_modified_mae'])
plt.title('Model MAE')
plt.ylabel('mae')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper right')
plt.show()

## Save parameters
with open(parameter_path, 'w') as f:
    yaml.dump(P, f)
