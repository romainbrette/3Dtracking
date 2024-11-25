'''
Makes a dataset from a movie or tiff folder.

Assumes the trajectories are in pixel.
TODO:
- load movies (not just tiffs)
- automatic focus determination
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

root = tk.Tk()
root.withdraw()

### Files and folders
movie_filename = filedialog.askopenfilename(initialdir=os.path.expanduser('~/Downloads/'), title='Choose a movie file')
traj_filename = filedialog.askopenfilename(initialdir=os.path.dirname(movie_filename), title='Choose a trajectory file')
path = filedialog.askdirectory(initialdir=os.path.dirname(traj_filename), title='Choose a dataset folder')

data = magic_load_trajectories(traj_filename)
img_path = os.path.join(path, 'images')
label_path = os.path.join(path, 'labels.csv')
parameter_path = os.path.join(path, 'labels.yaml')
if not os.path.exists(img_path):
    os.mkdir(img_path)

### Get image size
image = iio.imread(movie_filename)
width, height = image.shape[1], image.shape[0]

### Parameters
parameters = [('angle', 'Angle (°)', 19.2), # signed
              ('focus_point', 'Focus point (%)', 100), # x position where in focus
              ('pixel_size', 'Pixel size (um)', 5.),
              ('image_size', 'Image size (um)', 200),
              ('nimages', 'Number of images', 10000)
              ]
param_dialog = (ParametersDialog(title='Enter parameters', parameters=parameters))
P = param_dialog.value

pixel_size = P['pixel_size']
angle = P['angle']*np.pi/180.
P['image_size'] = (int(P['image_size']/pixel_size)//32)*32
image_size = P['image_size']
half_img_size = int(image_size/2)
P['focus_point'] = P['focus_point']/100*width

### Calculate the interval between frames
total_frames = data['frame'].nunique()
n_frames = int(P['nimages']/len(data)*total_frames)
frame_increment = int(total_frames/n_frames)

### Data frame
df = pd.DataFrame(columns=['filename', 'mean_z'])

### Iterate through frames
image_path = os.path.dirname(movie_filename)
files = [f for f in os.listdir(image_path) if f.endswith('.tiff') or f.endswith('.tif')]
files.sort()

n = 0
j = 0
for i in tqdm.tqdm(range(n_frames)):
    if n>=len(files):
        break
    image = iio.imread(os.path.join(image_path, files[n]))
    width, height = image.shape[1], image.shape[0]

    data_frame = data[data['frame'] == n]

    for _, row in data_frame.iterrows():
        # Make the window
        x0, y0 = row['x'], row['y']
        z = (x0 - P['focus_point']) * np.tan(P['angle'])  # mean z at the x position

        # Crop
        x1, y1 = x0-half_img_size, y0-half_img_size
        x2, y2 = x1+image_size, y1+image_size
        if (x1<0) or (y1<0) or (x2>width) or (y2>height): # skip if outside the image
            continue

        # Crop image
        # Apparently y=0 is the top
        snippet = image[int(y1):int(y2), int(x1):int(x2)]

        # Make the label file
        row = pd.DataFrame([{'filename' : 'im{:05d}.png'.format(j), 'mean_z' : z}])
        df = pd.concat([df, row], ignore_index=True)

        # Save image
        imageio.imwrite(os.path.join(img_path, 'im{:05d}.png'.format(j)), snippet)

        j += 1

    n += frame_increment

## Save labels
df.to_csv(label_path, index=False)

## Save parameters
with open(parameter_path, 'w') as f:
    yaml.dump(P, f)
