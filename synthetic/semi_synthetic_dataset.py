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
import albumentations as A

root = tk.Tk()
root.withdraw()

### Folders
path = filedialog.askdirectory(initialdir=os.path.expanduser('~/Downloads/'), message='Choose a dataset folder')
img_path = os.path.join(path, 'images')
label_path = os.path.join(path, 'labels.csv')
parameter_path = os.path.join(path, 'labels.yaml')
if not os.path.exists(img_path):
    os.mkdir(img_path)

image_path = filedialog.askdirectory(initialdir=path, message='Choose a base image folder')

### Parameters
parameters = [('width', 'Width (um)', 12000),
              ('depth', 'Depth (um)', 260),
              ('angle', 'Angle (°)', 19.2), # signed
              ('focus_point', 'Focus point (um)', 12000), # x position where in focus
              ('pixel_size', 'Pixel size (um)', 5.),
              ('frames', 'Number of images', 10000),
              ('image_size', 'Image size (um)', 200),
              ('blur', 'Blur factor', 0.0001) # proportionality factor between focal distance and blur size
              ]
param_dialog = (ParametersDialog(title='Enter parameters', parameters=parameters))
P = param_dialog.value

pixel_size = P['pixel_size']
angle = P['angle']*np.pi/180.
P['image_size'] = (int(P['image_size']/pixel_size)//32)*32
image_size = P['image_size']

### Load images
files = [f for f in os.listdir(image_path) if f.endswith('.tiff') or f.endswith('.tif') or f.endswith('.png')]
images = [iio.imread(os.path.join(image_path, f)) for f in tqdm.tqdm(files)]
image_size_original = images[0].shape[1] # assume square
if image_size_original!=image_size: # crop
    x1 = image_size_original//2 - image_size//2
    images = [I[x1:x1+image_size, x1:x1+image_size] for I in tqdm.tqdm(images)]

### Check whether background is white or black (assuming uint8)
if images[0].mean()<128: # black
    print('Black background')
    black_background = True
    #images = [255-image for image in images]
else:
    black_background = False

### Data frame
df = pd.DataFrame(columns=['filename', 'z', 'x', 'mean_z', 'sigma'])

## Blurred image
def random_image():
    image = images[random.randint(0, len(images)-1)]/255.
    # Random rotation
    angle = random.random()*360.
    return A.rotate(image, angle, border_mode=0, value=1.-black_background)

## Iterate
for i in tqdm.tqdm(np.arange(P['frames'])):
    ## Position
    x = P['width']*np.random.rand()
    mean_z = (x-P['focus_point'])*np.tan(angle) # mean z at the x position
    z = mean_z + (np.random.rand()-.5)*P['depth'] # z = 0 means in focus

    # Blurring
    sigma = abs(z)*P['blur']
    blurred_image = gaussian_filter(random_image(), sigma=sigma/pixel_size)

    # Make the label file
    row = pd.DataFrame(
        [{'filename': 'im{:06d}.png'.format(i), 'z': z, 'x' : x, 'mean_z' : mean_z, 'sigma' : sigma}])
    df = pd.concat([df, row], ignore_index=True)
    imageio.imwrite(os.path.join(img_path, 'im{:06d}.png'.format(i)), (blurred_image* 255).astype(np.uint8))

## Make a big image
big_image = np.ones((int(P['width']/pixel_size)+image_size, int(P['width']/pixel_size)+image_size))
density = 10/1e6 # 10/mm2
N = int(density*P['width']**2)
if black_background:
    big_image = 0.*big_image
for i in range(N):
    ## Position
    x = P['width']*np.random.rand()
    y = P['width']*np.random.rand()
    z = (x-P['focus_point'])*np.tan(angle) + (np.random.rand()-.5)*P['depth'] # z = 0 means in focus

    blurred_image = gaussian_filter(random_image(), sigma=abs(z)*P['blur']/pixel_size)
    if black_background:
        patch = np.where(blurred_image < .99, blurred_image,
                         big_image[int(y / pixel_size):int(y / pixel_size) + image_size,
                         int(x / pixel_size):int(x / pixel_size) + image_size])
    else:
        patch = np.where(blurred_image > .01, blurred_image,
                         big_image[int(y / pixel_size):int(y / pixel_size) + image_size,
                         int(x / pixel_size):int(x / pixel_size) + image_size])
    big_image[int(y / pixel_size):int(y / pixel_size) + image_size,
    int(x / pixel_size):int(x / pixel_size) + image_size] = patch
imageio.imwrite(os.path.join(path, 'big_image.png'), (big_image* 255).astype(np.uint8))

## Save labels
df.to_csv(label_path, index=False, float_format='%.2f') # two decimals

## Save parameters
with open(parameter_path, 'w') as f:
    yaml.dump(P, f)
