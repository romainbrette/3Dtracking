'''
Figure: validation error on the tilted slide used for training.
'''
from plotting import *
from pylab import *
import matplotlib
matplotlib.use('TkAgg')
from tkinter import filedialog
import tkinter as tk
import pandas as pd
import os
from gui.gui import *
import random

root = tk.Tk()
root.withdraw()

### Folders
label_path = filedialog.askopenfilename(initialdir=os.path.expanduser('~/Downloads/'), message='Choose a label file')
results_path = os.path.splitext(label_path)[0]+'_results.png'

### PIXEL SIZE
## Parameters
parameters = [('pixel_size', 'Pixel size (um)', 1.78),
              ('validation', 'Validation (last 20%)', True),
              ('detail', 'Detail', False)]
param_dialog = (ParametersDialog(title='Enter parameters', parameters=parameters))
P = param_dialog.value
pixel_size = P['pixel_size']

if P['detail']:
    figsize = (5, 5) # width, height
else:
    figsize = (5, 2.5) # width, height

## Load last 20% (validation part)
df = pd.read_csv(label_path)
if P['validation']:
    df = df.iloc[-len(df)//5:]

z_predict, mean_z = df['z_predict'].values*P['pixel_size'], df['mean_z'].values*P['pixel_size']

zmin, zmax = mean_z.min(), mean_z.max()

# Define the number of bins and bin edges
num_bins = 20
bin_edges = np.linspace(zmin, zmax, num_bins + 1)

## Error vs. mean z
# Digitize x_vals to find the bin each x belongs to
bin_indices = np.digitize(mean_z, bin_edges)

# Calculate the mean value in each bin
bin_means = [np.mean(z_predict[bin_indices == i]) for i in range(1, num_bins + 1)]
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2  # Midpoint of each bin for plotting

### Plotting
f1 = figure(1, figsize=figsize)
if P['detail']:
    ax1, ax2 = f1.subplots(2, 1)
    prepare_panel(ax1, ax2)
else:
    ax1 = f1.subplots()
    prepare_panel(ax1)

if P['validation']:
    ax1.scatter(mean_z, z_predict, alpha=0.05, s=4)
else:
    #sample = random.sample(range(len(mean_z)), len(mean_z)//5)
    ax1.scatter(mean_z, z_predict, alpha=0.01, s=4)
ax1.plot([zmin, zmax], [zmin, zmax], 'k--')
ax1.plot(bin_centers, bin_means, "r")
ax1.set_ylabel(r"z estimate ($\mu$m)")

if P['detail']:
    ax2.plot(bin_centers, [np.std(z_predict[bin_indices == i]) for i in range(1, num_bins + 1)], "b", label='s.d.')
    ax2.plot(bin_centers, [np.mean((z_predict[bin_indices == i]-mean_z[bin_indices == i])**2)**.5 for i in range(1, num_bins + 1)], "k", label='error')
    ax2.plot(bin_centers, [np.mean((z_predict[bin_indices == i]-mean_z[bin_indices == i])) for i in range(1, num_bins + 1)], "r", label='bias')
    ax2.plot(bin_centers, 0*bin_centers, "--k")
    ax2.set_ylabel(r'Error ($\mu$m)')
    ax2.set_xlabel(r'z ($\mu$m)')
    ax2.legend()
else:
    ax1.set_xlabel(r'z ($\mu$m)')

tight_layout()

savefig(results_path, dpi=300)

show()
