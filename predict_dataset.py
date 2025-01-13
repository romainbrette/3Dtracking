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
import tqdm
import yaml
import math

root = tk.Tk()
root.withdraw()

### Folders
path = filedialog.askdirectory(initialdir=os.path.expanduser('~/Downloads/'), message='Choose a dataset folder')
img_path = os.path.join(path, 'images')
label_path = os.path.join(path, 'labels.csv')
dataset_parameter_path = os.path.join(path, 'labels.yaml')

## Model
model_filename = filedialog.askdirectory(initialdir=path, message='Choose a model')

## Parameters
parameters = [('suffix', 'Suffix', '')]
param_dialog = (ParametersDialog(title='Enter parameters', parameters=parameters))
P = param_dialog.value

if P['suffix'] == '':
    output_path = os.path.join(path, 'labels_with_prediction.csv')
else:
    output_path = os.path.join(path, 'labels_with_prediction_'+P['suffix']+'.csv')

## Load data
df = pd.read_csv(label_path)

## Read dataset parameters
with open(dataset_parameter_path, 'r') as f:
    P_dataset = yaml.safe_load(f)
# Normalization factor
normalization = P_dataset.get('normalization', 1.)

# Extract filenames and labels
filenames = df['filename'].values

## Load and run model
model = load_model(model_filename, custom_objects={'modified_mae' : None, 'mean_abs_difference_metric':None, 'combined_loss':None})

# Define the function to load and preprocess images
def load_and_preprocess_image(filename):
    # Load the image
    image = tf.io.read_file(filename)
    image = tf.image.decode_png(image, channels=1)
    #image = tf.cast(image, tf.float32) * normalization
    image = tf.cast(image, tf.float32)
    mean_intensity = tf.reduce_mean(image)
    mean_intensity = tf.maximum(mean_intensity, 1e-8)
    image = image/mean_intensity

    #image = np.expand_dims(image, axis=0)
    return image

images = [load_and_preprocess_image(os.path.join(img_path, file)) for file in tqdm.tqdm(filenames)]

results = []
batch_size = 128
n_batch = math.ceil(len(images)/batch_size)
for i in tqdm.tqdm(range(n_batch), total=n_batch):
    results.extend(model.predict(np.array(images[i*batch_size:(i+1)*batch_size])))

df['z_predict'] = np.array(results)

print('MAE = ', np.mean(np.abs((df['z_predict'] - df['mean_z']).values)))

## Save
df.to_csv(output_path, index=False, float_format='%.2f')
