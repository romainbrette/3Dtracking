'''
Application: thigmotaxis and surface attraction
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
from scipy.ndimage import median_filter

dt = 1./20.
pixel_size = 1.78
refraction = 1.33

root = tk.Tk()
root.withdraw()  # Hide the main window
filename = filedialog.askopenfilename(initialdir=os.path.expanduser('~/Downloads/Deep learning movies/'), title='Choose a tracking file')

### Load FastTrack trajectories
data = magic_load_trajectories(filename)

#data = data[(data['z']<338.5) | (data['z']>339)]

# Convert pixels to um
data['x'] *= pixel_size
data['y'] *= pixel_size
data['z'] *= pixel_size*refraction # to compensate for air/water difference
data = filter_shape(data)

### Select all contiguous segments
segments = segments_from_table(data)

### Filter segments: longer than 1 s and spanning a distance of at least min_distance
min_distance = 0  # in um
min_duration = 1.
selected_segments = [segment for segment in segments if len(segment) > int(min_duration / dt) and \
                     ((segment['x'].max() - segment['x'].min()) ** 2 + (
                             segment['y'].max() - segment['y'].min()) ** 2) > min_distance ** 2]
print(len(selected_segments))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# ax = fig.add_subplot(111)
for segment in selected_segments[-10:]:
    x, y, z, t = segment['x'], segment['y'], segment['z'], segment['frame']*dt #- segment['x']*tan(7.2*np.pi/180)
    z = median_filter(z, size=11)

    # Plot the trajectory
    ax.plot(x, y, z, 'k')
    #t = segment['frame']*dt

## Swimming trajectories
# figure()
# for segment in selected_segments:
#     x, y, z = segment['x'], segment['y'], segment['z']/pixel_size #- segment['x']*tan(7.2*np.pi/180)
#     z = median_filter(z, size=11)
#     if ((z>200).any() & (z<200).any()):
#         plot(x, z)

data = pd.concat(selected_segments)

# calculate actual 3D speed
for segment in selected_segments:
    vx, vy = np.diff(segment['x']), np.diff(segment['y'])
    vz = np.diff(segment['z'])
    segment['vx'] = np.hstack([vx, nan])
    segment['vy'] = np.hstack([vy, nan])
    segment['vz'] = np.hstack([vz, nan])
    speed = (vx ** 2 + vy ** 2) ** .5
    speed_3D = (vx ** 2 + vy ** 2 + vz**2) ** .5
    segment['speed'] = np.hstack([speed, nan])
    segment['speed_3D'] = np.hstack([speed_3D, nan])

data = pd.concat(selected_segments)

figure()
hist(data['z'], 50)

figure()
subplot(211)
for segment in selected_segments[:10]:
    x, y, z, t = segment['x'], segment['y'], segment['z'], segment['frame']*dt
    #z = median_filter(z, size=11)
    #r = ((x-cx)**2 + (y-cy)**2)**.5
    plot(t, z)
#
#     # Plot the trajectory
#     #ax.plot(x, y, z)
#     #t = segment['frame']*dt
#     ax.plot(t, z)
subplot(212)
for segment in selected_segments[:10]:
    speed, t = segment['speed']/dt, segment['frame']*dt
    plot(t, speed)

figure()
#speed, z = data['speed_3D']/dt, data['z']
speed, z = data['speed']/dt, data['z']
scatter(z, speed, alpha=0.2, s=4)

figure()
x, y = data['x'], data['y']
subplot(211)
scatter(x, z, alpha=0.2, s=4)
subplot(212)
scatter(y, z, alpha=0.2, s=4)

## Distribution of attached cells
figure()
subplot(211)
hist(z[speed<50.], 50)
title("Attached cells (<50 um/s)")
subplot(212)
hist(z[speed>150.], 50)
title("Swimming cells (>150 um/s)")

plt.show()
