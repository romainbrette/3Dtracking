'''
Predict z on a movie (tiff folder).

IN PROGRESS
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
from movie.movie import *
from movie.cell_extraction import *

root = tk.Tk()
root.withdraw()

### Files and folders
movie_filename = filedialog.askopenfilename(initialdir=os.path.expanduser('~/Downloads/'), message='Choose a movie file')
traj_filename = filedialog.askopenfilename(initialdir=os.path.dirname(movie_filename), message='Choose a trajectory file')
name, ext = os.path.splitext(traj_filename)
output_trajectories = name+'_with_z.csv'

## Model
model_filename = filedialog.askdirectory(initialdir=os.path.dirname(movie_filename), message='Choose a model')

### Load the trained model
model = load_model(model_filename)
image_size = model.input_shape[1]
half_img_size = image_size//2

### Load trajectories
data = magic_load_trajectories(traj_filename)
data['z'] = np.nan

### Parameters
# parameters = [('pixel_size', 'Pixel size (um)', 5.)]
# param_dialog = (ParametersDialog(title='Enter parameters', parameters=parameters))
# P = param_dialog.value
# pixel_size = P['pixel_size']

### Open movie
image_path = os.path.dirname(movie_filename)
movie = MovieFolder(image_path, auto_invert=True)

### Get image size
image = movie.current_frame()
width, height = image.shape[1], image.shape[0]
n_frames = data['frame'].nunique()

### Iterate through frames
previous_position = 0
for image in tqdm.tqdm(movie.frames(), total=n_frames):
    data_frame = data[data['frame'] == previous_position]
    if len(data_frame)>0:
        snippets = extract_cells(image, data_frame, image_size, crop=True)
        normalization = 1./np.mean([np.mean(snippet) for snippet in snippets])
        predictions = model.predict(np.array([snippet*normalization for snippet in snippets]))
        data.loc[data['frame'] == previous_position, 'z'] = predictions
    previous_position = movie.position

print(data.head())

data.to_csv(output_trajectories)
