'''
Application: gravitaxis
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
from sklearn.mixture import GaussianMixture
from tracking.trajectory_analysis import *
from gui.gui import *
import yaml
import random
import seaborn as sns
from tracking.z_smoothing import *
from scipy.stats import binned_statistic_2d, binned_statistic

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

### Load FastTrack trajectories
data = magic_load_trajectories(filename)

## Convert pixels to um
data['x'] *= pixel_size
data['y'] *= pixel_size
data['z'] *= pixel_size*refraction # to compensate for air/water difference
#data = filter_shape(data)

## Retrack in 3D
data['z'] *= .05
data = norfair_track(data, distance_threshold=100, memory=10, delay=2, with_z=True, filtered=True)#, velocity=True)
data['z'] *= 20

zmin, zmax = data['z'].min(), data['z'].max()

### Select all contiguous segments
segments = segments_from_table(data)
print(len(segments))
print(np.mean([len(segment) for segment in segments]))

### Filter segments: longer than 1 s and spanning a distance of at least min_distance
min_distance = 1000  # in um
min_duration = 1.
selected_segments = [segment for segment in segments if len(segment) > int(min_duration / dt) and \
                     ((segment['x'].max() - segment['x'].min()) ** 2 + (
                             segment['y'].max() - segment['y'].min()) ** 2) > min_distance ** 2]
print(len(selected_segments))

### Calculate speed
data = pd.concat([calculate_speed(segment) for segment in selected_segments])

### Smoothness
BV = abs_variation(data)
P['mean_dz'] = float(BV)
print('Mean temporal variation of z:', BV, 'um')

### Bimodal fit
gmm = GaussianMixture(n_components=2, random_state=0)
gmm.fit(data['z'].values.reshape(-1, 1))

# Extract the parameters
means = gmm.means_.flatten()
stds = np.sqrt(gmm.covariances_).flatten()
dprime = np.abs(.5*(stds[0]+stds[1])/(means[1]-means[0]))
P['mu1'], P['mu2'] = float(means[0]), float(means[1])
P['std1'], P['std2'] = float(stds[0]), float(stds[1])
P['dprime'] = float(dprime)

print()
print('Means:', means)
print('Stds:', stds)
print('dprime:', dprime)
print()

## Distribution of z
figure()
z = data['z']
speed = data['speed_2D']/dt
subplot(311)
hist(z, 50)
title('All cells')
subplot(312)
hist(z[speed<50.], 50)
title("Attached cells (<50 um/s)")
subplot(313)
hist(z[speed>150.], 50)
xlabel('z (um)')
title("Swimming cells (>150 um/s)")
tight_layout()

## z vs. (x,y)
x, y = data['x'], data['y']
stat, x_edges, y_edges, _ = binned_statistic_2d(x, y, z, statistic='mean', bins=10)
# Create a meshgrid for plotting
x_centers = (x_edges[:-1] + x_edges[1:]) / 2
y_centers = (y_edges[:-1] + y_edges[1:]) / 2
X, Y = np.meshgrid(x_centers, y_centers)
# Plot
plt.figure(figsize=(6, 5))
plt.pcolormesh(X, Y, stat.T, cmap='viridis', shading='auto')
plt.colorbar(label='Mean Z value')

## z vs. x
bin_means, bin_edges, _ = binned_statistic(x, z, statistic='mean', bins=20)
# Compute bin centers
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
# Plot
plt.figure(figsize=(6, 4))
plt.plot(bin_centers, bin_means, marker='o', linestyle='-')

## z vs. t for a sample
#selected_segments = [trajectory for trajectory in trajectories_from_table(data) if trajectory['speed']]
selected_segments = trajectories_from_table(data)
figure()
annotated_segments = [(len(segment), segment) for segment in selected_segments]
annotated_segments.sort(reverse=True, key=lambda x:x[0])
pick = [segment for _,segment in annotated_segments[:15]]
#pick = random.sample(selected_segments, 15)
subplot(211)
for segment in pick:
    z, t = segment['z'], segment['frame']*dt
    #z = median_filter(z, size=11)
    plot(t, z)
ylabel('z (um)')
subplot(212)
for segment in pick:
    speed, t = segment['speed_2D']/dt, segment['frame']*dt
    plot(t, speed)
ylabel('Speed (um/s)')

## x vs. y with z in color
fig, ax = plt.subplots()
for segment in pick:
    x, y, z = segment['x'].values, segment['y'].values, segment['z'].values
    plot_colored_trajectory(ax, x, y, z, color_min=zmin, color_max=zmax)
ax.autoscale()
xlabel('x (um)')
ylabel('y (um)')

## Sample 3D trajectories
fig0 = plt.figure()
ax0= fig0.add_subplot(111, projection='3d')
fig1 = plt.figure()
ax1 = fig1.add_subplot(111, projection='3d')
fig2 = plt.figure()
ax2 = fig2.add_subplot(111)
# ax = fig.add_subplot(111)
for i, segment in enumerate(pick[:2]):
    print(i)
    x, y, z0, t = segment['x'].values, segment['y'].values, segment['z'].values, segment['frame'].values*dt
    n = 1500
    x, y, z0, t = x[:n], y[:n], z0[:n], t[:n]
    ax0.plot(x, y, z0, 'k')
    ax0.set_box_aspect([1,1,1])
    ax2.plot(t, z0, 'k')
    ## not sure the fancy approach works better
    #z = smooth_time_series(x, y, z0, lambda_smooth=100000., learning_rate=.1, iterations=50000)
    z = median_filter(z0, size=21)
    print(np.mean(np.abs(z-z0)))
    #z = median_filter(z, size=11)
    # Plot the trajectory
    ax1.plot(x, y, z, 'r')
    ax1.set_box_aspect([1,1,1])
    ax2.plot(t, z, 'r')

# figure()
# x, y = data['x'], data['y']
# subplot(211)
# scatter(x, z, alpha=0.2, s=4)
# subplot(212)
# scatter(y, z, alpha=0.2, s=4)

# figure()
# for segment in pick:
#     z, x = segment['z'], segment['x']
#     #z = median_filter(z, size=11)
#     plot(x, z)
#
# figure()
# scatter(data['x'], data['z'], alpha=0.05, s=4)

# trajectories = trajectories_from_table(data)
# swimming_segments = [segment for segment in trajectories if ((segment['z']>-150).any()) & ((segment['z']<-350).any())]
# annotated_segments = [(len(segment), segment) for segment in swimming_segments]
# annotated_segments.sort(reverse=True, key=lambda x:x[0])
# pick = [segment for _,segment in annotated_segments[:15]] # number 10 is interesting (ie index 9)

# if False:
#     for segment in pick[:10]:
#         z, x, speed, t = segment['z'].values, segment['x'].values, segment['speed'].values, segment['frame'].values*dt
#         #plot(speed, z)
#
#         # Create segments for the curve
#         points = np.array([speed, z]).T.reshape(-1, 1, 2)
#         segments = np.concatenate([points[:-1], points[1:]], axis=1)
#         # Normalize the color variable
#         norm = Normalize(vmin=t.min(), vmax=t.max())
#
#         # Create a LineCollection
#         lc = LineCollection(segments, cmap='rainbow', norm=norm)
#         lc.set_array(t)  # Set the variable controlling the color
#         lc.set_linewidth(2)  # Line width
#
#         # Plot the colored curve
#         fig, ax = plt.subplots() # figsize=(8, 5)
#         ax.add_collection(lc)
#         ax.autoscale()  # Adjust axes to fit the data
#         ax.set_xlim(speed.min(), speed.max())
#         ax.set_ylim(z.min(), z.max())
# elif True:
#     for segment in pick[7:8]:
#         z, x, speed, t = segment['z'].values, segment['x'].values, segment['speed'].values, segment['frame'].values * dt
#         # plot(speed, z)
#
#         # Create segments for the curve
#         points = np.array([t, z]).T.reshape(-1, 1, 2)
#         segments = np.concatenate([points[:-1], points[1:]], axis=1)
#         # Normalize the color variable
#         norm = Normalize(vmin=speed.min(), vmax=speed.max())
#
#         # Create a LineCollection
#         lc = LineCollection(segments, cmap='viridis', norm=norm)
#         lc.set_array(speed)  # Set the variable controlling the color
#         lc.set_linewidth(2)  # Line width
#
#         # Plot the colored curve
#         fig, ax = plt.subplots()  # figsize=(8, 5)
#         ax.add_collection(lc)
#         ax.autoscale()  # Adjust axes to fit the data
#         ax.set_xlim(t.min(), t.max())
#         ax.set_ylim(z.min(), z.max())
# elif False:
#     for segment in pick[:10]:
#         z, x, y, speed, t = segment['z'].values, segment['x'].values, segment['y'].values, segment['speed_3D'].values, segment['frame'].values * dt
#         # plot(speed, z)
#
#         # Create segments for the curve
#         points = np.array([x, y]).T.reshape(-1, 1, 2)
#         segments = np.concatenate([points[:-1], points[1:]], axis=1)
#         # Normalize the color variable
#         norm = Normalize(vmin=z.min(), vmax=z.max())
#
#         # Create a LineCollection
#         lc = LineCollection(segments, cmap='viridis', norm=norm)
#         lc.set_array(z)  # Set the variable controlling the color
#         lc.set_linewidth(2)  # Line width
#
#         # Plot the colored curve
#         fig, ax = plt.subplots()  # figsize=(8, 5)
#         ax.add_collection(lc)
#         ax.autoscale()  # Adjust axes to fit the data
#         ax.set_xlim(x.min(), x.max())
#         ax.set_ylim(y.min(), y.max())
# else:
#     for segment in pick[:10]:
#         z, x, y, speed, t = segment['z'].values, segment['x'].values, segment['y'].values, segment['speed_3D'].values, segment['frame'].values * dt
#         # plot(speed, z)
#
#         fig = plt.figure()
#         ax = fig.add_subplot(111, projection='3d')
#         ax.plot(x, y, z, 'k')


with open(output_file, 'w') as f:
    yaml.dump(P, f)

plt.show()
