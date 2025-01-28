'''
Figure: test error on the recordings with fixed depth.
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

figsize = (3,3)

root = tk.Tk()
root.withdraw()

### Folders
label_path = filedialog.askopenfilename(initialdir=os.path.expanduser('~/Downloads/'), message='Choose a label file')
results_path = os.path.splitext(label_path)[0]+'_results.png'

### PIXEL SIZE
parameters = [('pixel_size', 'Pixel size (um)', 1.78)]
param_dialog = (ParametersDialog(title='Enter parameters', parameters=parameters))
P = param_dialog.value
pixel_size = P['pixel_size']

## Load
df = pd.read_csv(label_path)
df = df.sort_values(by='mean_z')
df['z_predict'], df['mean_z'] = df['z_predict']*pixel_size, df['mean_z']*pixel_size

### Plotting
f1 = figure(1, figsize=figsize)
ax1 = f1.subplots()
prepare_panel(ax1)

# mean_estimate = df.groupby('mean_z')['z_predict'].mean()
# error = df.groupby('mean_z')['z_predict'].std()
# mean_z = df.groupby('mean_z')['mean_z'].mean()

result = df.groupby('mean_z')['z_predict'].agg(
    mean_estimate='mean',
    mean_abs_diff=lambda group: (group - group.mean()).abs().mean()
).reset_index()
mean_estimate = result['mean_estimate'].values
error = result['mean_abs_diff'].values
mean_z = result['mean_z'].values

m, M = df['mean_z'].min(), df['mean_z'].max()
ax1.plot([m,M], [m,M], 'k--')
ax1.plot(mean_z, mean_estimate, color='blue')
ax1.fill_between(mean_z, mean_estimate-error, mean_estimate+error, color='blue', alpha=0.3)
ax1.set_ylabel(r'z estimate ($\mu$m)')
ax1.set_xlabel(r'z ($\mu$m)')

tight_layout()

savefig(results_path, dpi=300)

show()
