'''
Train a network to estimate z from Paramecium image.

There was an issue with brightness_range: it can be done on int images (255), not normalized.

Roughly 45 s / epoch with the first model; the tuned model is horribly slow (400 s).

It seems that changing the network structure has little effect.

- maybe stratified dataset with eccentricity?
- MobileNetV3?
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
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input, GlobalAveragePooling2D, Reshape, Lambda
import random
import tkinter as tk
from tkinter import filedialog
from tensorflow.keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
from keras.utils import plot_model

load_checkpoint = False
save_checkpoint = True
batch_size = 64 # seems slightly faster than 32
number_of_files = None
epochs = 500

root = tk.Tk()
root.withdraw()  # Hide the main window
path = filedialog.askdirectory(initialdir=os.path.expanduser('~/Downloads/Deep learning movies/'), title='Choose a folder')
root.destroy()

img_path = os.path.join(path, 'dataset', 'images')
label_path = os.path.join(path, 'dataset', 'labels.csv')
checkpoint_filename = os.path.join(path,'best_z_estimation_augmentation_simple.tf')

## Read data
df = pd.read_csv(label_path)

# Extract filenames and labels
filenames = df['filename'].values
labels = df['x'].values # * np.tan(7.2*np.pi/180) # what is the angle here?
n = len(filenames)

# Selection
if number_of_files is not None:
    files = random.sample(range(n), number_of_files)
    filenames = filenames[files]
    labels = labels[files]
    n = len(filenames)

# Define the function to load and preprocess images
def load_and_preprocess_image(filename):
    # Load the image
    image = tf.io.read_file(filename)
    image = tf.image.decode_png(image, channels=1)
    #image = tf.cast(image, tf.float32) / 255.0  # Normalize to [0, 1]
    # Compute the maximum value in the image
    #max_val = tf.reduce_max(image)
    # Avoid division by zero
    #image = tf.math.divide_no_nan(image, max_val)
    return image

images = np.array([load_and_preprocess_image(os.path.join(img_path,file)) for file in filenames])
labels = np.array(labels)

#train_images, val_images, train_labels, val_labels = train_test_split(images, labels, test_size=0.2, random_state=42)
n_split = int(n*4./5)
train_images, val_images, train_labels, val_labels = images[:n_split], images[n_split:], labels[:n_split], labels[n_split:]

train_datagen = ImageDataGenerator(
    rescale=1./255,
    #rotation_range=360, ## probably not that useful
    #width_shift_range=0.2,
    #height_shift_range=0.2,
    #shear_range=0.2,
    #zoom_range=0.2, ## possibly useful, but would also distort the z cue
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
    epochs=epochs,
    callbacks=callbacks
)

# Evaluate the model
loss, mae = model.evaluate(val_generator)
print(f'Validation loss: {loss}, Validation MAE: {mae}')

model.save(os.path.join(path,'z_estimation.tf'))

# Plot
plt.plot(history.history['mae'])
plt.plot(history.history['val_mae'])
plt.title('Model MAE')
plt.ylabel('mae')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper right')
plt.show()
