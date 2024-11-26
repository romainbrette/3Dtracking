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
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
from keras.utils import plot_model
from tkinter import filedialog
from gui.gui import *
import yaml

### Folders
path = filedialog.askdirectory(initialdir=os.path.expanduser('~/Downloads/'), title='Choose a folder')
img_path = os.path.join(path, 'dataset', 'images')
label_path = os.path.join(path, 'dataset', 'labels.csv')
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
n = len(filenames)

## Load images
def load_and_preprocess_image(filename):
    # Load the image
    image = tf.io.read_file(filename)
    image = tf.image.decode_png(image, channels=1)
    return image
images = np.array([load_and_preprocess_image(os.path.join(img_path,file)) for file in filenames])

## Make training and validation sets
n_split = int(n*(1-validation_ratio))
train_images, val_images, train_labels, val_labels = images[:n_split], images[n_split:], labels[:n_split], labels[n_split:]

train_dataset = create_dataset(train_filenames, train_labels)
val_dataset = create_dataset(val_filenames, val_labels)

keras.utils.split_dataset(
    dataset, left_size=None, right_size=None, shuffle=False, seed=None
)

train_datagen = ImageDataGenerator(
    rescale=1./255,
    #rotation_range=360, ## probably not that useful
    brightness_range=[0.5, 2.],
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='constant', # filling with black (background removed) # not useful actually if no rotations
    cval=0
    #fill_mode='nearest'
)
val_datagen = ImageDataGenerator(
    rescale=1./255,
    brightness_range=[0.5, 2.] # maybe not at validation
)

def plot_augmented_images(datagen, image, num_images=9):
    plt.figure(figsize=(10, 10))
    for i in range(num_images):
        # Generate an augmented image
        batch = next(datagen.flow(image, batch_size=1))
        print(batch[0].max(), batch[0].mean())
        augmented_image = (batch[0]*255.).astype('uint8')

        # Plot the image
        plt.subplot(3, 3, i + 1)
        plt.imshow(augmented_image, cmap='gray')
        plt.axis('off')
    plt.show()

# Plot augmented images
#image = images[0]
#image = image.reshape((1,) + image.shape)
#plot_augmented_images(train_datagen, image)
#exit(0)

# Create training and validation generators
train_generator = train_datagen.flow(train_images, train_labels, batch_size=batch_size)
val_generator = val_datagen.flow(val_images, val_labels, batch_size=batch_size)

shape = images[0].shape
print("Image shape:", shape)

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

# model = Sequential([Conv2D(32, (3, 3), activation='relu', input_shape=shape),
#                     MaxPooling2D((2, 2))])
# for _ in range(4):
#     model.add(Conv2D(32, (3, 3), activation='relu')) # should I progressively increase this number?
#     model.add(MaxPooling2D((2, 2)))
# model.add(Flatten())
# #model.add(Dense(64, activation='relu'))
# model.add(Dense(1))

# model = Sequential([
#  Conv2D(32, (5, 5), activation='relu', input_shape=shape), # was 32; (3,3)
#  GlobalAveragePooling2D(),
#  Dense(32, activation='relu'), # perhaps try tunable swish
#  Dense(1)
# ])

model.summary()

# Load weights from the checkpoint
if load_checkpoint:
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
    train_generator,
    validation_data=val_generator,
    epochs=P['epochs'],
    callbacks=callbacks
)

# Evaluate the model
loss, mae = model.evaluate(val_generator)
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
