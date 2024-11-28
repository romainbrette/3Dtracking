'''
Makes a synthetic dataset with defocused cell images to estimate z.

TODO:
- take actual focused images as basis (with rotations maybe etc.)
'''
import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.draw import ellipse
import os
import imageio
import pandas as pd
from tkinter import filedialog
from gui.gui import *
import yaml
import tkinter as tk
import tqdm

root = tk.Tk()
root.withdraw()

### Folders
path = filedialog.askdirectory(initialdir=os.path.expanduser('~/Downloads/'), title='Choose a dataset folder')
img_path = os.path.join(path, 'images')
label_path = os.path.join(path, 'labels.csv')
parameter_path = os.path.join(path, 'labels.yaml')
if not os.path.exists(img_path):
    os.mkdir(img_path)

### Parameters
parameters = [('width', 'Width (um)', 12000),
              ('depth', 'Depth (um)', 260),
              ('angle', 'Angle (°)', 19.2), # signed
              ('focus_point', 'Focus point (um)', 12000), # x position where in focus
              ('pixel_size', 'Pixel size (um)', 5.),
              ('cell_length', 'Cell length (um)', 120),
              ('cell_width', 'Cell width  (um)', 30),
              ('image_size', 'Image size (um)', 200),
              ('frames', 'Number of images', 10000),
              ('blur', 'Blur factor', 0.001) # proportionality factor between focal distance and blur size
              ]
param_dialog = (ParametersDialog(title='Enter parameters', parameters=parameters))
P = param_dialog.value

pixel_size = P['pixel_size']
angle = P['angle']*np.pi/180.
P['image_size'] = (int(P['image_size']/pixel_size)//32)*32
image_size = P['image_size']
cell_half_width = int(P['cell_width']//(2*pixel_size)) # in pixels
cell_half_length = int(P['cell_length']//(2*pixel_size))

### Data frame
df = pd.DataFrame(columns=['filename', 'z', 'x', 'mean_z'])

## Blurred image
def random_image():
    ## Cell image
    cell_image = np.ones((image_size, image_size))
    orientation = np.random.rand()*np.pi
    length = int(cell_half_width + (cell_half_length-cell_half_width)*np.random.rand())
    # Black perimeter
    rr, cc = ellipse(image_size//2, image_size//2, cell_half_width, length, shape=cell_image.shape, rotation=orientation)
    cell_image[rr, cc] = 0
    # Grey inside
    rr, cc = ellipse(image_size//2, image_size//2, cell_half_width-1, length-1, rotation=orientation, shape=cell_image.shape)
    cell_image[rr, cc] = 0.5
    return cell_image

## Iterate
for i in tqdm.tqdm(np.arange(P['frames'])):
    ## Position
    x = P['width']*np.random.rand()
    mean_z = (x-P['focus_point'])*np.tan(angle) # mean z at the x position
    z = mean_z + (np.random.rand()-.5)*P['depth'] # z = 0 means in focus

    # Blurring
    blurred_image = gaussian_filter(random_image(), sigma=abs(z)*P['blur'])

    # Make the label file
    row = pd.DataFrame(
        [{'filename': 'im{:05d}.png'.format(i), 'z': z, 'x' : x, 'mean_z' : mean_z}])
    df = pd.concat([df, row], ignore_index=True)
    imageio.imwrite(os.path.join(img_path, 'im{:05d}.png'.format(i)), (blurred_image* 255).astype(np.uint8))

## Make a big image
big_image = np.ones((int(P['width']/pixel_size)+image_size, int(P['width']/pixel_size)+image_size))
for i in range(50):
    ## Position
    x = P['width']*np.random.rand()
    y = P['width']*np.random.rand()
    z = (x-P['focus_point'])*np.tan(angle) + (np.random.rand()-.5)*P['depth'] # z = 0 means in focus

    blurred_image = gaussian_filter(random_image(), sigma=abs(z)*P['blur'])
    big_image[int(y/pixel_size):int(y/pixel_size)+image_size, int(x/pixel_size):int(x/pixel_size)+image_size] = blurred_image
imageio.imwrite(os.path.join(path, 'big_image.png'), (big_image* 255).astype(np.uint8))

## Save labels
df.to_csv(label_path, index=False, float_format='%.2f') # two decimals

## Save parameters
with open(parameter_path, 'w') as f:
    yaml.dump(P, f)
