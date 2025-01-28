'''
Figure: thigmotaxis
'''
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
from tracking.trajectory_analysis import *
from gui.gui import *
import seaborn as sns
import matplotlib.pyplot as plt

refraction = 1.33

root = tk.Tk()
root.withdraw()  # Hide the main window
filename = filedialog.askopenfilename(initialdir=os.path.expanduser('~/Downloads/Deep learning movies/'), title='Choose a tracking file')
name, ext = os.path.splitext(filename)
results_path = os.path.splitext(filename)[0]+'_results.png'

### Parameters
parameters = [('pixel_size', 'Pixel size (um)', 1.78),
              ('fps', 'Frame rate (Hz)', 20.)
              ]
param_dialog = (ParametersDialog(title='Enter parameters', parameters=parameters))
P = param_dialog.value
pixel_size = P['pixel_size']
dt = 1./P['fps']

### Load FastTrack trajectories
data = magic_load_trajectories(filename)

## Convert pixels to um
data['x'] *= pixel_size
data['y'] *= pixel_size
data['z'] *= pixel_size*refraction # to compensate for air/water difference
data = filter_shape(data)

## Track using z
data = norfair_track(data, distance_threshold=200, memory=4, delay=2, velocity=True)

### Select all contiguous segments
segments = segments_from_table(data)

### Calculate speed
data['speed'] = data['speed_kalman']/dt

sns.jointplot(
    x=data['z'],
    y=data['speed'],
    kind="scatter",
    alpha=0.15,
    marker='.',
    s=4,
    marginal_kws=dict(bins=40, fill=True),
    color="black",
    height=3
)
# # Slide boundaries
_, ymax = plt.gca().get_ylim()
plot([0, 0], [0, ymax], 'k--')
plot([-520, -520], [0, ymax], 'k--')
xlim(-600, 50)
ylim(0, 800)
xlabel(r'z ($\mu$m)')
ylabel(r'Speed ($\mu$m/s)')

savefig(results_path, dpi=300)

show()
