'''
Evaluate a trained model on a dataset.
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
import yaml
from sklearn.feature_selection import mutual_info_regression
from scipy.stats import pearsonr

#pixel_size = 1.78

root = tk.Tk()
root.withdraw()

### Folders
label_path = filedialog.askopenfilename(initialdir=os.path.expanduser('~/Downloads/'), message='Choose a label file')
results_path = os.path.splitext(label_path)[0]+'_results.png'
info_path = os.path.splitext(label_path)[0]+'_results.yaml'
info = {}

df = pd.read_csv(label_path)
#df['z_predict'] = df['z_predict']/1.78 #pixel_size
z_predict, mean_z = df['z_predict'].values, df['mean_z'].values

## Error
RMSE = ((z_predict-mean_z)**2).mean()**.5
print("Error vs. mean_z:", RMSE)
info["RMSE"] = RMSE

MAE = np.mean(np.abs(z_predict-mean_z))
print("Mean absolute error:", MAE)
info['MAE'] = MAE

## Calculate mutual information
MI = mutual_info_regression(z_predict, mean_z, discrete_features=[False])
print("Mutual information:", MI, "bits")
info["MI"] = MI

## Linear correlation
correlation, _ = pearsonr(z_predict, mean_z)
info['r'] = correlation

zmin, zmax = mean_z.min(), mean_z.max()

# Define the number of bins and bin edges
num_bins = 100
bin_edges = np.linspace(zmin, zmax, num_bins + 1)

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
plot(bin_centers, [np.mean((z_predict[bin_indices == i]-mean_z[bin_indices == i])) for i in range(1, num_bins + 1)], "b", label='bias')
plot(bin_centers, 0*bin_centers, "--b")
ylabel('Error (um)')
#ylim(bottom=0)
legend()

subplot(414)
plot(bin_centers, [np.sum([bin_indices == i]) for i in range(1, num_bins + 1)], "r")
xlabel('Mean z (um)')
ylabel('Density')
ylim(bottom=0)

savefig(results_path)
with open(info_path, 'w') as f:
    yaml.dump(info, f)

show()
