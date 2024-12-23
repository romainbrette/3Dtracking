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
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Rescaling, Dense, GlobalAveragePooling2D, BatchNormalization, LeakyReLU, Dropout, ReLU, GlobalMaxPool2D
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
from tensorflow.keras.applications import ResNet50, EfficientNetV2B0
from tensorflow.keras.layers import Input, Dense, Conv2D, Flatten, MaxPooling2D, Concatenate
from tensorflow.keras.models import Model

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
checkpoint_filename = os.path.join(path, 'best_z_'+P['filename_suffix']+'.tf')
parameter_path = os.path.join(path, 'training_'+P['filename_suffix']+'.yaml')
history_path = os.path.join(path, 'history_'+P['filename_suffix']+'.csv')
model_filename = os.path.join(path, 'model_'+P['filename_suffix']+'.txt')
batch_size = 128
validation_ratio = 0.2 # proportion of images used for validation

## Read data
df = pd.read_csv(label_path)

## Extract filenames and labels
filenames = df['filename'].values
if P['predict_sigma']:
    if 'sigma' in df:
        labels = df['sigma'].values
    else:
        labels = df['z'].values
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
    #img = tf.image.grayscale_to_rgb(img)
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
print(images.min(), images.max(), np.std(images))

## Image moments layer
class ImageMomentsLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(ImageMomentsLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # No trainable weights
        pass

    def call(self, inputs):
        """
        Inputs: Tensor of shape (batch_size, height, width, channels)
        Outputs: Tensor of shape (batch_size, 10), where 10 corresponds
                 to moments up to order 3 (m00, m01, m10, m11, m20, etc.)
        """
        # Ensure grayscale images
        #if inputs.shape[-1] > 1:
        #    inputs = tf.image.rgb_to_grayscale(inputs)

        # Remove the channel dimension if present
        if inputs.shape.rank == 4:  # Check if inputs have a channel dimension
            inputs = tf.squeeze(inputs, axis=-1)  # Remove channel dimension

        # Convert to binary (thresholding at 0.5 normalized range)
        #binary = tf.cast(inputs > 0.5, tf.float32)

        # Image dimensions
        batch_size = tf.shape(inputs)[0]
        height = tf.shape(inputs)[1]
        width = tf.shape(inputs)[2]

        # Generate coordinate grids
        x_coords = tf.linspace(-1., 1., width)
        y_coords = tf.linspace(-1., 1., height)
        x_coords, y_coords = tf.meshgrid(x_coords, y_coords)

        # Expand dimensions to broadcast over the batch
        x_coords = tf.expand_dims(x_coords, axis=0)  # Shape (1, height, width)
        y_coords = tf.expand_dims(y_coords, axis=0)  # Shape (1, height, width)

        # Broadcast to match input shape
        x_coords = tf.tile(x_coords, [batch_size, 1, 1])  # Shape (batch_size, height, width)
        y_coords = tf.tile(y_coords, [batch_size, 1, 1])  # Shape (batch_size, height, width)

        # Moments calculation
        m00 = tf.reduce_sum(inputs, axis=[1, 2])
        m10 = tf.reduce_sum(inputs * x_coords, axis=[1, 2])
        m01 = tf.reduce_sum(inputs * y_coords, axis=[1, 2])
        m11 = tf.reduce_sum(inputs * x_coords * y_coords, axis=[1, 2])
        m20 = tf.reduce_sum(inputs * tf.square(x_coords), axis=[1, 2])
        m02 = tf.reduce_sum(inputs * tf.square(y_coords), axis=[1, 2])
        m21 = tf.reduce_sum(inputs * x_coords * tf.square(y_coords), axis=[1, 2])
        m12 = tf.reduce_sum(inputs * y_coords * tf.square(x_coords), axis=[1, 2])
        m30 = tf.reduce_sum(inputs * tf.pow(x_coords, 3), axis=[1, 2])
        m03 = tf.reduce_sum(inputs * tf.pow(y_coords, 3), axis=[1, 2])

        # Concatenate moments into a single feature vector
        moments = tf.stack([m00, m10, m01, m11, m20, m02, m21, m12, m30, m03], axis=1)

        return moments

    def compute_output_shape(self, input_shape):
        return (input_shape[0], 10)

## Load weights and model from the checkpoint
if P['load_checkpoint']:
    print('Loading previous model')
    #model.load_weights(checkpoint_filename)
    model = tf.keras.models.load_model(checkpoint_filename)
elif False:
    # Load a pretrained model
    base_model = EfficientNetV2B0(weights='imagenet', include_top=False, input_shape=shape)

    # Build regression model
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(128, activation='relu'),
        Dense(1, activation='linear')  # Single continuous output
    ])
elif False:
    # Moments-based model with convolution too
    # Input layer for image data
    input_layer = Input(shape=shape)  # Example size of 64x64 grayscale image

    # Branch 1: Image moments
    moments_layer = ImageMomentsLayer()(input_layer)

    # Branch 2: Conv2D layers
    conv_layer = Conv2D(32, (3, 3), activation='relu', padding='same')(input_layer)
    #pool_layer = MaxPooling2D((2, 2))(conv_layer)
    #flat_layer = Flatten()(pool_layer)
    flat_layer = GlobalAveragePooling2D()(conv_layer)

    # Combine both branches
    combined = Concatenate()([moments_layer, flat_layer])

    # Dense layers for regression
    dense_layer = Dense(64, activation='relu')(combined)
    dense_layer2 = Dense(64, activation='relu')(dense_layer)
    output_layer = Dense(1, activation='linear')(dense_layer2)

    # Build model
    model = Model(inputs=input_layer, outputs=output_layer)
elif False:
    # Moments-based model
    model = Sequential([
        ImageMomentsLayer(input_shape=shape),
        Dense(128, activation='relu'),
        Dense(1, activation='linear')  # Single continuous output
    ])
elif False:
    # Load a pretrained model
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=shape)

    # Build regression model
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(128, activation='relu'),
        Dense(1, activation='linear')  # Single continuous output
    ])
elif True:
    model = Sequential([
        Flatten(input_shape=shape),
        Dense(64, activation='leaky_relu'),
        BatchNormalization(),
        Dense(64, activation='leaky_relu'),
        BatchNormalization(),
        Dense(1)
    ])
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
        Conv2D(16, (3, 3), kernel_initializer='he_normal', input_shape=shape, activation="leaky_relu"),
        BatchNormalization(),
        #MaxPooling2D((2, 2)),

        # Conv2D(64, (3, 3), kernel_initializer='he_normal', activation="relu"),
        # BatchNormalization(),
        # MaxPooling2D((2, 2)),
        #
        # Conv2D(128, (3, 3), kernel_initializer='he_normal', activation="relu"),
        # BatchNormalization(),
        # MaxPooling2D((2, 2)),

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
        GlobalMaxPool2D(),
        #Flatten(),
        #Dropout(0.5),
        Dense(64, activation='leaky_relu', kernel_initializer='he_normal'),
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

reduce_lr = ReduceLROnPlateau(monitor='loss',
                              factor=0.5,         # Reduce learning rate by half
                              patience=2, # this must be adapted to the dataset size
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
print(parameter_path, P['filename_suffix'])
with open(parameter_path, 'w') as f:
    yaml.dump(P, f)
