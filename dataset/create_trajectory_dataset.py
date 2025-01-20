'''
Makes a dataset from a movie or tiff folder.
The label file is ordered by trajectories, and a binary number marks the endpoint of a trajectory with 0 (vs. 1).

** Assumes the trajectories are in pixel. **

TODO:
- remove cells with close neighbors
- load movies (not just tiffs)
- automatic focus determination
- deal with trajectories in um
'''
import numpy as np
import os
import pandas as pd
from tkinter import filedialog
from gui.gui import *
import yaml
import tkinter as tk
from tracking.load_tracking import *
from tracking.trajectory_analysis import *
import imageio
import tqdm
from movie.movie import *
from movie.cell_extraction import *

root = tk.Tk()
root.withdraw()

### Files and folders
movie_filename = filedialog.askopenfilename(initialdir=os.path.expanduser('~/Downloads/'), message='Choose a movie file')
traj_filename = filedialog.askopenfilename(initialdir=os.path.dirname(movie_filename), message='Choose a trajectory file')
path = filedialog.askdirectory(initialdir=os.path.dirname(traj_filename), message='Choose a dataset folder')
background_path = filedialog.askopenfilename(initialdir=os.path.dirname(movie_filename), message='Choose a background image')

data = magic_load_trajectories(traj_filename)
img_path = os.path.join(path, 'images')
label_path = os.path.join(path, 'labels.csv')
parameter_path = os.path.join(path, 'labels.yaml')
if not os.path.exists(img_path):
    os.mkdir(img_path)

### Parameters
parameters = [('angle', 'Angle (°)', 19.2), # signed
              ('in_pixel', 'Trajectory in pixel', True),
              ('focus_point', 'Focus point (%)', 100), # x position where in focus
              ('pixel_size', 'Pixel size (um)', 5.),
              ('image_size', 'Image size (um)', 200),
              ('nimages', 'Number of images', 0),
              ('min_distance', 'Minimum trajectory size (um)', 300.)
              ]
param_dialog = (ParametersDialog(title='Enter parameters', parameters=parameters))
P = param_dialog.value

pixel_size = P['pixel_size']
angle = P['angle']*np.pi/180.
P['image_size'] = (int(P['image_size']/pixel_size)//32)*32
image_size = P['image_size']
half_img_size = int(image_size/2)
min_distance = P['min_distance'] # trajectories must span sufficient distance to be included

### Put data in pixels
if not P['in_pixel']:
    data['x'] /= pixel_size
    data['y'] /= pixel_size

### Calculate the interval between frames
total_frames = data['frame'].nunique()
if P['nimages'] == 0: # all images
    frame_increment = 1
    n_frames = total_frames
else:
    n_frames = int(P['nimages']/len(data)*total_frames)
    frame_increment = int(total_frames/n_frames)

### Background normalization
if background_path:
    background = imageio.imread(background_path)
    normalization = np.mean(background)
else:
    normalization = 1.


### Open movie
image_path = os.path.dirname(movie_filename)
movie = MovieFolder(image_path, step=frame_increment, auto_invert=True)

### Get image size
image = movie.current_frame()
width, height = image.shape[1], image.shape[0]
P['focus_point'] = P['focus_point']/100*width

### Data frame
df = pd.DataFrame(columns=['filename', 'mean_z'])
# could be vectorized:
#mean_z = (df['x'] - P['focus_point']) * np.tan(P['angle'])  # mean z at the x position
#mean_z = mean_z.iloc[::frame_increment]

### Exclude cells close to the border
data = data[(data['x']>half_img_size) & (data['y']>half_img_size) & \
        (data['x']+half_img_size<width) & (data['y']+half_img_size<height)]

### Exclude stationary trajectories
traj = segments_from_table(data)
selected_segments = [segment for segment in traj if \
                     ((segment['x'].max()-segment['x'].min())**2 + (segment['y'].max()-segment['y'].min())**2)>(min_distance/pixel_size)**2]
data = pd.concat(selected_segments)

### Make images
j = 0
previous_position = 0
rows = {}
intensities = []
for image in tqdm.tqdm(movie.frames(), total=n_frames):
    data_frame = data[data['frame'] == previous_position]
    snippets = extract_cells(image, data_frame, image_size, crop=True, borders=False) # remove border cells because they give away x and therefore z
    intensities.extend([np.mean(snippet) for snippet in snippets])

    i = 0 # this is snippet number
    for row_index, row in data_frame.iterrows():
        if snippets[i] is not None:
            j += 1 # this is image number
            z = (row['x'] - P['focus_point']) * np.tan(angle)  # mean z at the x position

            # Make the label file
            rows[row_index] = {'filename' : 'im{:06d}.png'.format(j), 'mean_z' : z}

            # Save image
            imageio.imwrite(os.path.join(img_path, 'im{:06d}.png'.format(j)), snippets[i])

        i += 1

    previous_position = movie.position

### Make the label table
filenames = []
z = []
mask = []
for segment in selected_segments:
    rows_indexes = segment.index.values
    for i in range(len(rows_indexes)):
        row = rows_indexes[i]
        filenames.append(rows[row]['filename'])
        z.append(rows[row]['mean_z'])
    mask.extend([1]*(len(rows_indexes)-1)+[0])
df = pd.DataFrame({'filename': filenames,
                   'mean_z': z,
                   'mask': mask})

if not P['in_pixel']: # save in the same unit as in the trajectory file
    df['mean_z'] *= pixel_size

## Save labels
df.to_csv(label_path, index=False)

## Save parameters
P['normalization'] = float(1./np.mean(intensities))
with open(parameter_path, 'w') as f:
    yaml.dump(P, f)
