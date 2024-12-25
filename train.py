'''
Train a network to estimate z from Paramecium image.

TODO:
- Metrics: on both z and z_mean
'''
import pandas as pd
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, GlobalAveragePooling2D, BatchNormalization, LeakyReLU, Lambda, Dropout
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
from tensorflow.keras.regularizers import l2
import sys
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from augmentation.augmentation import *

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
              ('filename_suffix', 'Filename suffix', ''),
              ('background_subtracted', 'Background subtracted', True), # if it is background subtracted, the background is constant # could be done automatically
              ('min_scaling', 'Minimum intensity scaling', 1.),
              ('max_scaling', 'Maximum intensity scaling', 1.)
              ]
param_dialog = (ParametersDialog(title='Enter parameters', parameters=parameters))
P = param_dialog.value
save_checkpoint = True
checkpoint_filename = os.path.join(path,'best_z_'+P['filename_suffix']+'.tf')
parameter_path = os.path.join(path, 'training_'+P['filename_suffix']+'.yaml')
history_path = os.path.join(path, 'history_'+P['filename_suffix']+'.csv')
model_filename = os.path.join(path, 'model_'+P['filename_suffix']+'.txt')
dataset_parameter_path = os.path.join(path, 'labels.yaml')
batch_size = 128
validation_ratio = 0.2 # proportion of images used for validation

## Read data
df = pd.read_csv(label_path)

## Shuffle
df = df.sample(frac=1).reset_index(drop=True)

## Read dataset parameters
with open(dataset_parameter_path, 'r') as f:
    P_dataset = yaml.safe_load(f)
# Normalization factor
normalization = P_dataset.get('normalization', 1.)

## Extract filenames and labels
filenames = df['filename'].values
if P['predict_sigma']:
    if 'sigma' in df:
        labels = df['sigma'].values
    else:
        labels = df['z'].values
else:
    try:
        labels = df['mean_z'].values
    except KeyError:
        labels = df['z'].values
filenames = [os.path.join(img_path, name) for name in filenames]
n = len(filenames)

## Range of output values
span = float(labels.max()-labels.max())

## Load images

# Create a mapping function for loading and preprocessing images
def load_image(filename, label):
    # Load the image from the file path
    img = tf.io.read_file(filename)
    img = tf.image.decode_png(img, channels=1)
    img = tf.cast(img, dtype=tf.float32)
    img = img*normalization # normalize so as to have mean image = 1.0
    #img = img / 255.0  # Normalize pixel values to [0, 1]
    #img = tf.image.grayscale_to_rgb(img)
    return img, label

## Get image shape
image, _ = load_image(filenames[0], None)
image = np.array(image)
shape = image.shape
print("Image shape:", shape)

# Create a tf.data.Dataset from filenames and labels
dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
dataset = dataset.map(lambda filename, label: load_image(filename, label), num_parallel_calls=tf.data.AUTOTUNE)

## Make training and validation sets
train_dataset, val_dataset = tf.keras.utils.split_dataset(dataset, right_size=validation_ratio, shuffle=False)

## Data augmentation

if P['background_subtracted']:
    intensity_scaling = RandomIntensityScaling(P['min_scaling'], P['max_scaling'])
else:
    intensity_scaling = layers.RandomBrightness(factor=[P['min_scaling'], P['max_scaling']], value_range=[0., 1.])
data_augmentation = tf.keras.Sequential([
    #layers.Rescaling(1./255),
    #layers.RandomFlip("horizontal_and_vertical"), ### This crashes with the GPU!!
    intensity_scaling
    #layers.RandomRotation(1., fill_mode="constant", fill_value=1.-black_background*1.)  ### This crashes with the GPU!!
])
#just_rescaling = layers.Rescaling(1./255)

train_dataset = train_dataset.map(lambda x, y: (data_augmentation(x), y), num_parallel_calls=AUTOTUNE)
#val_dataset = val_dataset.map(lambda x, y: (just_rescaling(x), y), num_parallel_calls=AUTOTUNE)
val_dataset = val_dataset.map(lambda x, y: (data_augmentation(x), y), num_parallel_calls=AUTOTUNE)

## Prepare
train_dataset = train_dataset.shuffle(buffer_size=1000).batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
val_dataset = val_dataset.shuffle(buffer_size=1000).batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
#train_dataset = train_dataset.batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
#val_dataset = val_dataset.batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)

## Load weights and model from the checkpoint
if P['load_checkpoint']:
    print('Loading previous model')
    #model.load_weights(checkpoint_filename)
    model = tf.keras.models.load_model(checkpoint_filename)
elif False:
    model = Sequential([
        Flatten(input_shape=shape),
        Dense(64, activation='leaky_relu'),
        BatchNormalization(),
        Dense(64, activation='leaky_relu'),
        BatchNormalization(),
        Dense(1)
    ])
elif False:
    model = Sequential([ # tuned model, but I'm not sure, final receptive fields are too small
        Conv2D(75, (5, 5), activation='leaky_relu', input_shape=shape),
        MaxPooling2D((2, 2)),
        Conv2D(87, (5, 5), activation='leaky_relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(161, activation='leaky_relu'),
        Dense(1)
    ])
elif True:
    model = Sequential([  # tuned model, but I'm not sure, final receptive fields are too small
        Conv2D(32, (3, 3), activation='leaky_relu', input_shape=shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='leaky_relu'),
        #MaxPooling2D((2, 2)),
        GlobalAveragePooling2D(),
        Flatten(),
        Dense(128, activation='leaky_relu'),
        #Lambda(lambda x: x * span),
        Dense(1)
    ])
elif False: ## I think this one generalizes better
    model = Sequential([ # tuned model, but I'm not sure, final receptive fields are too small
        Conv2D(75, (5, 5), activation='leaky_relu', input_shape=shape),
        MaxPooling2D((2, 2)),
        Conv2D(87, (5, 5), activation='leaky_relu'),
        MaxPooling2D((2, 2)),
        GlobalAveragePooling2D(),
        Flatten(),
        Dense(161, activation='leaky_relu'),
        Dense(1)
    ])
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
else:
    model = Sequential([
        Conv2D(32, (3, 3), kernel_initializer='he_normal', input_shape=shape),
        BatchNormalization(),  # Doesn't seem to work
        LeakyReLU(),  # Activation après BN
        MaxPooling2D((2, 2)),

        Conv2D(64, (3, 3), kernel_initializer='he_normal'),
        BatchNormalization(),
        LeakyReLU(),  # Activation après BN
        MaxPooling2D((2, 2)),

        #GlobalAveragePooling2D(),
        Flatten(),
        #Dropout(0.5),
        Dense(128, activation='relu', kernel_initializer='he_normal'),
        Dense(1)
    ])

model.summary()
with open(model_filename, "w") as f:
    sys.stdout = f  # Redirect output to the file
    model.summary()
    sys.stdout = sys.__stdout__  # Reset to default

## Compile the model
model.compile(optimizer='adam', # default learning_rate .001
              loss='mean_squared_error', metrics=['mae'])
#model.compile(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.0005), # default learning_rate .001
#              loss='mean_squared_error', metrics=['mae'])

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
loss, mae = model.evaluate(val_dataset)
print(f'Validation loss: {loss}, Validation MAE: {mae}')
P['loss'] = loss
P['mae'] = mae
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
plt.title('Model MAE')
plt.ylabel('mae')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper right')
plt.show()

## Save parameters
with open(parameter_path, 'w') as f:
    yaml.dump(P, f)
