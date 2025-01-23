'''
Predict z on a movie (tiff folder).
'''
import numpy as np
import os
from tkinter import filedialog
from gui.gui import *
import yaml
import tkinter as tk
from tracking.load_tracking import *
import tqdm
from tensorflow.keras.models import load_model
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
model = load_model(model_filename, custom_objects={'modified_mae' : None, 'mean_abs_difference_metric':None, 'combined_loss':None})
image_size = model.input_shape[1]
half_img_size = image_size//2

### Load trajectories
data = magic_load_trajectories(traj_filename)
data['z'] = np.nan

### Parameters
parameters = [('normalize', 'Intensity normalization', False)]
param_dialog = (ParametersDialog(title='Enter parameters', parameters=parameters))
P = param_dialog.value

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
        if P['normalize']:
            predictions = model.predict(np.array([snippet / (np.mean(snippet) + 1e-8) for snippet in snippets]))
        else:
            predictions = model.predict(np.array(snippets))
        #predictions = model.predict(np.array([snippet*normalization for snippet in snippets]))
        data.loc[data['frame'] == previous_position, 'z'] = predictions
    previous_position = movie.position

print(data.head())

data.to_csv(output_trajectories)
