'''
Predict z on a movie (tiff folder).

IN PROGRESS

TODO: padding when images are near the border
'''
import numpy as np
import os
import pandas as pd
from tkinter import filedialog
from gui.gui import *
import yaml
import tkinter as tk
from tracking.load_tracking import *
import imageio
import imageio.v3 as iio
import tqdm
from tensorflow.keras.models import load_model
import tensorflow as tf

root = tk.Tk()
root.withdraw()

### Files and folders
movie_filename = filedialog.askopenfilename(initialdir=os.path.expanduser('~/Downloads/'), message='Choose a movie file')
traj_filename = filedialog.askopenfilename(initialdir=os.path.dirname(movie_filename), message='Choose a trajectory file')
name, ext = os.path.splitext(traj_filename)
output_trajectories = name+'_with_z'+ext

## Model
model_filename = filedialog.askdirectory(initialdir=os.path.dirname(movie_filename), message='Choose a model')

### Load trajectories
data = magic_load_trajectories(traj_filename)
data['z_predict'] = np.nan

### Get image size
image = iio.imread(movie_filename)
width, height = image.shape[1], image.shape[0]

### Parameters
parameters = [('pixel_size', 'Pixel size (um)', 5.)]
param_dialog = (ParametersDialog(title='Enter parameters', parameters=parameters))
P = param_dialog.value
pixel_size = P['pixel_size']

### Load the trained model
#options = tf.saved_model.LoadOptions(experimental_io_device='/job:localhost')
#model = load_model(model_filename, options=options)
model = load_model(model_filename)
image_size = model.input_shape[0]
half_img_size = int(image_size/2)

### Iterate through frames
image_path = os.path.dirname(movie_filename)
files = [f for f in os.listdir(image_path) if f.endswith('.tiff') or f.endswith('.tif')]
files.sort()

for n, file in enumerate(tqdm.tqdm(files)):
    image = iio.imread(os.path.join(image_path, files[n]))
    width, height = image.shape[1], image.shape[0]

    data_frame = data[data['frame'] == n]

    snippets = []
    for _, row in data_frame.iterrows():
        # Make the window
        x0, y0 = row['x'], row['y']

        # Crop
        x1, y1 = x0-half_img_size, y0-half_img_size
        x2, y2 = x1+image_size, y1+image_size
        if (x1 < 0) or (y1 < 0) or (x2 > width) or (y2 > height):  # skip if outside the image
            # snippet = np.zeros((img_size, img_size))
            ## this could be better, with padding
            snippet = None
        else:
            # Crop image
            # Apparently y=0 is the top
            snippet = image[int(y1):int(y2), int(x1):int(x2)]

        ## Apply model
        snippets.append(snippet)
        # snippet = np.expand_dims(snippet, axis=0)  # Add batch dimension
        # prediction = model.predict(snippet)
        # z = prediction[0][0]
        # print(z)
    full_snippets = [snippet for snippet in snippets if snippet is not None]
    which_ones = [snippet is not None for snippet in snippets]
    full_predictions = model.predict(np.array(full_snippets)).flatten()
    predictions = np.ones(len(snippets)) * np.nan
    predictions[which_ones] = full_predictions
    data.loc[data['frame'] == n, 'z'] = predictions

print(data.head())

data.to_csv(output_trajectories)
