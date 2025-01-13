'''
Produces a movie with z prediction written next to the cell.
'''
from tracking import *
import os
import numpy as np
from tkinter import filedialog
from movie.movie import *
from tkinter import filedialog
from gui.gui import *
import imageio

### Files and folders
movie_filename = filedialog.askopenfilename(initialdir=os.path.expanduser('~/Downloads/'), message='Choose a movie file')
traj_filename = filedialog.askopenfilename(initialdir=os.path.dirname(movie_filename), message='Choose a trajectory file')
name, ext = os.path.splitext(traj_filename)
movie_out = name+'.mp4'

### Load trajectories
data = magic_load_trajectories(traj_filename)

### Parameters
parameters = [('pixel_size', 'Pixel size (um)', 5.),
              ('fps', 'FPS (Hz)', 20.)]
param_dialog = (ParametersDialog(title='Enter parameters', parameters=parameters))
P = param_dialog.value
pixel_size = P['pixel_size']
fps = P['fps']

#data = data[data['frame']*dt<60.]
#data['angle'] = -data['angle']

### Open movie
image_path = os.path.dirname(movie_filename)
movie = MovieFolder(image_path, auto_invert=True)

write_movie_with_tracking(movie, movie_out, data, pixel_size=1, quality=5, radius=100, crop=True)

writer = imageio.get_writer(movie_out, fps=fps, quality=5)
font = ImageFont.truetype("Arial.ttf", 30)
nframes = int(table['frame'].max())
width = int((table['x'].max() + radius) / pixel_size)
height = int((table['y'].max() + radius) / pixel_size)
for n in range(int(table['frame'].min()), nframes + 1):
    if (n % 100) == 0:
        print(n, "/", nframes + 1)
    frame = np.ones((height, width), dtype=np.uint8) * 255
    image = ellipse_cells(frame, table[table['frame'] == n], pixel_size=pixel_size, normalize=False, filled=filled)
    if 'z' in table.columns:  # write z information
        pil_image = Image.fromarray(image)
        draw = ImageDraw.Draw(pil_image)
        for row in table[table['frame'] == n].itertuples():
            draw.text((int((row.x + radius) // pixel_size),
                       int((row.y / pixel_size))), str(int(row.z)), fill="red", font=font)
        # Convert back to a NumPy array
        image = np.array(pil_image)

    writer.append_data(image)
writer.close()

movie.close()
