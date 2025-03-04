'''
Produces a movie with z estimation written next to the cell.
'''
from tracking import *
import os
import numpy as np
from tkinter import filedialog
from movie.movie import *
from tkinter import filedialog
from gui.gui import *
from PIL import Image, ImageFont, ImageDraw
import tqdm
from skimage import color
import tkinter as tk
import imageio.v2 as iio

root = tk.Tk()
root.withdraw()

### Files and folders
movie_filename = filedialog.askopenfilename(initialdir=os.path.expanduser('~/Downloads/'), message='Choose a movie file')
traj_filename = filedialog.askopenfilename(initialdir=os.path.dirname(movie_filename), message='Choose a trajectory file')
name, ext = os.path.splitext(traj_filename)
movie_out = name+'.mp4'

### Load trajectories
data = magic_load_trajectories(traj_filename)

### Parameters
parameters = [('pixel_size', 'Pixel size (um)', 1.78),
              ('in_pixel', 'Trajectory in pixel', True),
              ('fps', 'FPS (Hz)', 20.),
              ('text_shift', 'Text shift (um)', 200),
              ('font_size', 'Font size (pixel)', 30),
              ('truncate', 'Truncation (s)', 1e6)]
param_dialog = (ParametersDialog(title='Enter parameters', parameters=parameters))
P = param_dialog.value
if P['in_pixel']:
    pixel_size = 1.
    z_factor = P['pixel_size']*1.33 ## air-water factor
else:
    pixel_size = P['pixel_size']
    z_factor = 1.33
fps = P['fps']
text_shift = int(P['text_shift']/P['pixel_size'])
font_size= int(P['font_size'])
print(movie_out, fps)

data = data[data['frame']*1/fps<P['truncate']]

### Open movie
if movie_filename.endswith('.zip'):
    movie = MovieZip(movie_filename, auto_invert=True, gray=True)
else:
    image_path = os.path.dirname(movie_filename)
    movie = MovieFolder(image_path, auto_invert=True)

### Write movie
writer = iio.get_writer(movie_out, fps=fps, quality=5)
font = ImageFont.truetype("Arial.ttf", font_size)
nframes = int(data['frame'].max())
for image in tqdm.tqdm(movie.frames(), total=nframes):
    n = movie.position - 1
    image = color.gray2rgb(np.iinfo(image.dtype).max-image)
    pil_image = Image.fromarray(image)
    draw = ImageDraw.Draw(pil_image)
    for row in data[data['frame'] == n].itertuples():
        xt = int(row.x // pixel_size) - text_shift
        if xt<0:
            xt = int(row.x // pixel_size) + text_shift
        draw.text((xt, int((row.y / pixel_size))), str(int(row.z*z_factor)), fill="red", font=font)
    # Convert back to a NumPy array
    image = np.array(pil_image)
    writer.append_data(image)
    if n >= nframes:
        break
writer.close()

movie.close()
