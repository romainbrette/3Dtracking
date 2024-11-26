'''
Makes a synthetic dataset with defocused cell images to estimate z.
Using actual focused images as basis.

4x: 4900 um
0.5x: 26640 um (whole field)
'''
import numpy as np
from scipy.ndimage import gaussian_filter
import os
import pandas as pd
from tkinter import filedialog
from gui.gui import *
import yaml
import tkinter as tk
import imageio
import imageio.v3 as iio
import tqdm
import random

root = tk.Tk()
root.withdraw()

### Folders
path = filedialog.askdirectory(initialdir=os.path.expanduser('~/Downloads/'), title='Choose a dataset folder')
img_path = os.path.join(path, 'images')
label_path = os.path.join(path, 'labels.csv')
parameter_path = os.path.join(path, 'labels.yaml')
if not os.path.exists(img_path):
    os.mkdir(img_path)

image_path = filedialog.askdirectory(initialdir=path, title='Choose an image folder')

### Parameters
parameters = [('width', 'Width (um)', 12000),
              ('depth', 'Depth (um)', 260),
              ('angle', 'Angle (°)', 19.2), # signed
              ('focus_point', 'Focus point (um)', 12000), # x position where in focus
              ('pixel_size', 'Pixel size (um)', 5.),
              ('frames', 'Number of images', 10000),
              ('blur', 'Blur factor', 0.001) # proportionality factor between focal distance and blur size
              ]
param_dialog = (ParametersDialog(title='Enter parameters', parameters=parameters))
P = param_dialog.value

pixel_size = P['pixel_size']
angle = P['angle']*np.pi/180.

### Load images
files = [f for f in os.listdir(image_path) if f.endswith('.tiff') or f.endswith('.tif') or f.endswith('.png')]
images = [iio.imread(os.path.join(image_path, f)) for f in tqdm.tqdm(files)]
image_size = images[0].shape[1] # assume square

### Check whether background is white or black (assuming uint8)
if images[0].mean()<128: # black
    print('Black background')
    black_background = True
    #images = [255-image for image in images]
else:
    black_background = False

### Data frame
df = pd.DataFrame(columns=['filename', 'z', 'x', 'mean_z'])

## Blurred image
def random_image():
    return images[random.randint(0, len(images)-1)]/255.

## Iterate
for i in tqdm.tqdm(np.arange(P['frames'])):
    ## Position
    x = P['width']*np.random.rand()
    mean_z = (x-P['focus_point'])*np.tan(P['angle']) # mean z at the x position
    z = mean_z + 2*(np.random.rand()-.5)*P['depth'] # z = 0 means in focus

    # Blurring
    blurred_image = gaussian_filter(random_image(), sigma=abs(z)*P['blur'])

    # Make the label file
    row = pd.DataFrame(
        [{'filename': 'im{:05d}.png'.format(i), 'z': z, 'x' : x, 'mean_z' : mean_z}])
    df = pd.concat([df, row], ignore_index=True)
    imageio.imwrite(os.path.join(img_path, 'im{:05d}.png'.format(i)), (blurred_image* 255).astype(np.uint8))

## Make a big image
big_image = np.ones((int(P['width']/pixel_size)+image_size, int(P['width']/pixel_size)+image_size))
if black_background:
    big_image = 0.*big_image
for i in range(50):
    ## Position
    x = P['width']*np.random.rand()
    y = P['width']*np.random.rand()
    z = (x-P['focus_point'])*np.tan(P['angle']) + 2*(np.random.rand()-.5)*P['depth'] # z = 0 means in focus

    blurred_image = gaussian_filter(random_image(), sigma=abs(z)*P['blur'])
    big_image[int(y/pixel_size):int(y/pixel_size)+image_size, int(x/pixel_size):int(x/pixel_size)+image_size] = blurred_image
imageio.imwrite(os.path.join(path, 'big_image.png'), (big_image* 255).astype(np.uint8))

## Save labels
df.to_csv(label_path, index=False, float_format='%.2f') # two decimals

## Save parameters
with open(parameter_path, 'w') as f:
    yaml.dump(P, f)
