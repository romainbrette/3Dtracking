'''
Makes a synthetic dataset with defocused cell images to estimate z.

TODO: produce a sample image with many cells
'''
import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.draw import ellipse
import os
import imageio
import pandas as pd
from tkinter import filedialog
from gui.gui import *

angle = 19.2*np.pi/180.
width = 12000 # in um
depth = 520
image_size = 32
half_img_size = image_size//2
pixel_size = 5
cell_width = 15//pixel_size # half
cell_height = 60//pixel_size
n_frames = 10000

parameters = [('width', 'Width (um)', 12000),
              ('depth', 'Depth (um)', 260),
              ('angle', 'Angle (°)', 19.2), # signed
              ('focus_point', 'Focus point (um)', 100), # x position where in focus
              ('pixel_size', 'Pixel size (um)', 5),
              ('cell_length', 'Cell length (um)', 120),
              ('cell_width', 'Cell width  (um)', 30),
              ('image_size', 'Image size (um)', 200),
              ('frames', 'Number of images', 10000),
              ('blur', 'Blur factor', 0.001) # proportionality factor between focal distance and blur size
              ]
path = filedialog.askdirectory(initialdir=os.path.expanduser('~/Downloads/'), title='Choose a folder')
P = ParametersDialog(title='Enter parameters', parameters=parameters).value

exit(0)
sigma_factor = 1/1000

## Folders
img_path = os.path.join(path, 'dataset', 'images')
label_path = os.path.join(path, 'dataset', 'labels.csv')
if not os.path.exists(os.path.join(path, 'dataset')):
    os.mkdir(os.path.join(path, 'dataset'))
if not os.path.exists(img_path):
    os.mkdir(img_path)

### Data frame
df = pd.DataFrame(columns=['filename', 'z', 'x'])

## Iterate
zmax = width * np.tan(angle) + depth
for i in range(n_frames):
    ## Position
    x = width*np.random.rand()
    z = x*np.tan(angle) + np.random.rand()*depth
    focal_position = zmax-z

    ## Cell image
    cell_image = np.ones((image_size, image_size))
    orientation = np.random.rand()*np.pi
    length = int(cell_width + (cell_height-cell_width)*np.random.rand())
    # Black perimeter
    rr, cc = ellipse(image_size//2, image_size//2, cell_width, length, shape=cell_image.shape, rotation=orientation)
    cell_image[rr, cc] = 0
    # Grey inside
    rr, cc = ellipse(image_size//2, image_size//2, cell_width-1, length-1, rotation=orientation, shape=cell_image.shape)
    cell_image[rr, cc] = 0.5

    ## Blurring
    blurred_image = gaussian_filter(cell_image, sigma=focal_position*sigma_factor)

    # Make the label file
    row = pd.DataFrame(
        [{'filename': 'im{:05d}.png'.format(i), 'z': z, 'x' : x}])
    df = pd.concat([df, row], ignore_index=True)
    imageio.imwrite(os.path.join(img_path, 'im{:05d}.png'.format(i)), (blurred_image* 255).astype(np.uint8))

## Save labels
df.to_csv(label_path, index=False)
