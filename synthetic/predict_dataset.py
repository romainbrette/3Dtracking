'''
Predict z on an image dataset.
'''
import os
import tensorflow as tf
from tkinter import filedialog
import tkinter as tk
from gui.gui import *
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model

root = tk.Tk()
root.withdraw()

### Folders
path = filedialog.askdirectory(initialdir=os.path.expanduser('~/Downloads/'), message='Choose a dataset folder')
img_path = os.path.join(path, 'images')
label_path = os.path.join(path, 'labels.csv')
output_path = os.path.join(path, 'labels_with_prediction.csv')

## Model
model_filename = filedialog.askdirectory(initialdir=path, message='Choose a model')

## Load data
df = pd.read_csv(label_path)

# Extract filenames and labels
filenames = df['filename'].values

# Define the function to load and preprocess images
def load_and_preprocess_image(filename):
    # Load the image
    image = tf.io.read_file(filename)
    image = tf.image.decode_png(image, channels=1)
    image = tf.cast(image, tf.float32) / 255.0
    return image

images = np.array([load_and_preprocess_image(os.path.join(img_path,file)) for file in filenames])
shape = images[0].shape

## Load and run model
model = load_model(model_filename)
df['z_predict'] = model.predict(images)

## Save
df.to_csv(output_path, index=False, float_format='%.2f')
