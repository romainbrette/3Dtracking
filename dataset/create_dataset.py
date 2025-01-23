'''
Makes a dataset from a movie or tiff folder.

TODO:
- remove cells with close neighbors?
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
from tracking.trajectory_analysis import *
import imageio
import tqdm
from movie.movie import *
from movie.cell_extraction import *
import zipfile
import io

root = tk.Tk()
root.withdraw()

### Files and folders
movie_filename = filedialog.askopenfilename(initialdir=os.path.expanduser('~/Downloads/'), message='Choose a movie file')
traj_filename = filedialog.askopenfilename(initialdir=os.path.dirname(movie_filename), message='Choose a trajectory file')
path = filedialog.askdirectory(initialdir=os.path.dirname(traj_filename), message='Choose a dataset folder')

data = magic_load_trajectories(traj_filename)
img_path = os.path.join(path, 'images')
label_path = os.path.join(path, 'labels.csv')
parameter_path = os.path.join(path, 'labels.yaml')

### Parameters
parameters = [('angle', 'Angle (°)', 19.2), # signed
              ('in_pixel', 'Trajectory in pixel', True),
              ('focus_point', 'Focus point (%)', 100), # x position where in focus
              ('pixel_size', 'Pixel size (um)', 5.),
              ('image_size', 'Image size (um)', 200),
              ('nimages', 'Number of images', 0),
              ('min_distance', 'Minimum trajectory size (um)', 300.),
              ('zip', 'Zip', True)
              ]
param_dialog = (ParametersDialog(title='Enter parameters', parameters=parameters))
P = param_dialog.value
pixel_size = P['pixel_size']
angle = P['angle']*np.pi/180.
P['image_size'] = (int(P['image_size']/pixel_size)//32)*32
image_size = P['image_size']
half_img_size = int(image_size/2)
min_distance = P['min_distance'] # trajectories must span sufficient distance to be included

if (not P['zip']) & (not os.path.exists(img_path)):
    os.mkdir(img_path)

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

### Open movie
if movie_filename.endswith('.zip'):
    movie = MovieZip(movie_filename, step=frame_increment, auto_invert=True, gray=True)
else:
    image_path = os.path.dirname(movie_filename)
    movie = MovieFolder(image_path, step=frame_increment, auto_invert=True, gray=True)

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

### Exclude cells that are too big or too small (cell pairs)
data = filter_shape(data, length=(10.*pixel_size, 200.*pixel_size), width=(8.*pixel_size, 80.*pixel_size))

### Exclude stationary trajectories
traj = trajectories_from_table(data)
selected_segments = [segment for segment in traj if \
                     ((segment['x'].max()-segment['x'].min())**2 + (segment['y'].max()-segment['y'].min())**2)>(min_distance/pixel_size)**2]
data = pd.concat(selected_segments)

if P['zip']:
    zip_ref = zipfile.ZipFile(img_path+'.zip', mode='w', compression=zipfile.ZIP_DEFLATED)
j = 0
intensities = []
previous_position = 0
for image in tqdm.tqdm(movie.frames(), total=n_frames):
    data_frame = data[data['frame'] == previous_position]
    snippets = extract_cells(image, data_frame, image_size, crop=True)
    intensities.extend([np.mean(snippet) for snippet in snippets])

    i = 0 # this is snippet number
    for _, row in data_frame.iterrows():
        j += 1 # this is image number
        z = (row['x'] - P['focus_point']) * np.tan(angle)  # mean z at the x position

        snippet = snippets[i]
        row = pd.DataFrame([{'filename' : 'im{:06d}.png'.format(j), 'mean_z' : z}])

        # Make the label file
        df = pd.concat([df, row], ignore_index=True)

        # Save image
        filename = 'im{:06d}.png'.format(j)
        if P['zip']:
            image_bytes = io.BytesIO()
            imageio.imwrite(image_bytes, snippet, format='png')
            image_bytes.seek(0)  # Move the cursor to the start of the BytesIO object
            zip_ref.writestr(filename, image_bytes.read())  # Add image as 'image1.png'
        else:
            imageio.imwrite(os.path.join(img_path, filename, snippet))

        i += 1

    previous_position = movie.position
if P['zip']:
    zip_ref.close()

if not P['in_pixel']: # save in the same unit as in the trajectory file
    df['mean_z'] *= pixel_size

P['normalization'] = float(1./np.mean(intensities))

## Save labels
df.to_csv(label_path, index=False)

## Save parameters
with open(parameter_path, 'w') as f:
    yaml.dump(P, f)
