'''
Test prediction results on a synthetic dataset.

IN PROGRESS
'''
import os
from tkinter import filedialog
import tkinter as tk
from gui.gui import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
from pylab import *

root = tk.Tk()
root.withdraw()

### Folders
label_path = filedialog.askopenfilename(initialdir=os.path.expanduser('~/Downloads/'), message='Choose a label file')

df = pd.read_csv(label_path)
z, z_predict, mean_z = df['z'].values, df['z_predict'].values, df['mean_z'].values

print("Error vs. z:", ((z_predict-z)**2).mean()**.5)
print("Error vs. mean_z:", ((z_predict-mean_z)**2).mean()**.5)

zmin, zmax = z.min(), z.max()

# Define the number of bins and bin edges
num_bins = 100
bin_edges = np.linspace(zmin, zmax, num_bins + 1)

# Digitize x_vals to find the bin each x belongs to
bin_indices = np.digitize(z, bin_edges)

# Calculate the mean y value in each bin
bin_means = [np.mean(z_predict[bin_indices == i]) for i in range(1, num_bins + 1)]
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2  # Midpoint of each bin for plotting

## Error vs. true z
figure()
subplot(211)
plot(z, z_predict, '.k')
plot([zmin, zmax], [zmin, zmax], 'b')
plot(bin_centers, bin_means, "r")
ylabel("Estimate of z (um)")

subplot(413)
plot(bin_centers, [np.std(z_predict[bin_indices == i]) for i in range(1, num_bins + 1)], "r", label='s.d.')
#plot(bin_centers, [np.mean(np.abs(z_predict[bin_indices == i]-z[bin_indices == i])) for i in range(1, num_bins + 1)], "k", label='error')
plot(bin_centers, [np.mean((z_predict[bin_indices == i]-z[bin_indices == i])**2)**.5 for i in range(1, num_bins + 1)], "k", label='error')
ylabel('Error (um)')
ylim(bottom=0)
legend()

subplot(414)
plot(bin_centers, [np.sum([bin_indices == i]) for i in range(1, num_bins + 1)], "r")
xlabel('z (um)')
ylabel('Density')
ylim(bottom=0)

## Error vs. mean z
# Digitize x_vals to find the bin each x belongs to
bin_indices = np.digitize(mean_z, bin_edges)

# Calculate the mean y value in each bin
bin_means = [np.mean(z_predict[bin_indices == i]) for i in range(1, num_bins + 1)]
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2  # Midpoint of each bin for plotting

figure()
subplot(211)
plot(mean_z, z_predict, '.k')
plot([zmin, zmax], [zmin, zmax], 'b')
plot(bin_centers, bin_means, "r")
ylabel("Estimate of z (um)")

subplot(413)
plot(bin_centers, [np.std(z_predict[bin_indices == i]) for i in range(1, num_bins + 1)], "r", label='s.d.')
#plot(bin_centers, [np.mean(np.abs(z_predict[bin_indices == i]-mean_z[bin_indices == i])) for i in range(1, num_bins + 1)], "k", label='error')
plot(bin_centers, [np.mean((z_predict[bin_indices == i]-mean_z[bin_indices == i])**2)**.5 for i in range(1, num_bins + 1)], "k", label='error')
ylabel('Error (um)')
ylim(bottom=0)
legend()

subplot(414)
plot(bin_centers, [np.sum([bin_indices == i]) for i in range(1, num_bins + 1)], "r")
xlabel('Mean z (um)')
ylabel('Density')
ylim(bottom=0)

show()
