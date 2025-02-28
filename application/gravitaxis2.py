'''
Application: gravitaxis
'''
import numpy as np
from pylab import *
from tracking import *
from optparse import OptionParser
import matplotlib.pyplot as plt
from matplotlib.widgets import SpanSelector
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import tkinter as tk
from tkinter import filedialog
import matplotlib
matplotlib.use('TkAgg')
from scipy.ndimage import median_filter
from sklearn.mixture import GaussianMixture
from tracking.trajectory_analysis import *
from gui.gui import *
import yaml
import random
import seaborn as sns
from tracking.z_smoothing import *

refraction = 1.33

root = tk.Tk()
root.withdraw()  # Hide the main window
filename = filedialog.askopenfilename(initialdir=os.path.expanduser('~/Downloads/Deep learning movies/'), title='Choose a tracking file')
name, ext = os.path.splitext(filename)
output_file = name+'_eval.yaml'

### Parameters
parameters = [('pixel_size', 'Pixel size (um)', 1.),
              ('fps', 'Frame rate (Hz)', 20.)
              ]
param_dialog = (ParametersDialog(title='Enter parameters', parameters=parameters))
P = param_dialog.value
pixel_size = P['pixel_size']
dt = 1./P['fps']

### Load trajectories
data = magic_load_trajectories(filename)

## Convert pixels to um
data['x'] *= pixel_size
data['y'] *= pixel_size
data['z'] *= pixel_size*refraction # to compensate for air/water difference
#data = filter_shape(data)

#data = norfair_track(data, distance_threshold=500, memory=10, delay=0)#, velocity=True)

### Select all contiguous segments
segments = segments_from_table(data)
print(len(segments))
print(np.mean([len(segment) for segment in segments]))

### Calculate vertical angle
theta, z, dtheta = [], [], []
for segment in segments:
    if len(segment) > 50:
        vx, vy = np.diff(segment['x']), np.diff(segment['y'])
        speed_2D = (vx**2 + vy**2)**.5
        speed_2D = median_filter(speed_2D, size=15)
        vz = np.diff(segment['z'])
        vz = median_filter(vz, size=51)
        angle = np.arctan(vz/speed_2D) # 0 is horizontal, pi/2 is upward
        theta.extend(list(angle[:-1]))
        dtheta.extend(list(np.diff(angle)))
theta = np.array(theta).flatten()
dtheta = np.array(dtheta).flatten()/dt

figure()
scatter(theta, dtheta, s=1)

plt.show()
