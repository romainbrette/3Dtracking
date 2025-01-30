'''
Show the ladder effect when the model is trained on a discrete set of z positions.
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
import seaborn as sns
import matplotlib.pyplot as plt

root = tk.Tk()
root.withdraw()

### Folders
label_path = filedialog.askopenfilename(initialdir=os.path.expanduser('~/Downloads/'), message='Choose a label file')
results_path = os.path.splitext(label_path)[0]+'_results.png'

### PIXEL SIZE
## Parameters
parameters = [('pixel_size', 'Pixel size (um)', 1.78),
              ('min_z', 'Minimum z (um)', -500.)]
param_dialog = (ParametersDialog(title='Enter parameters', parameters=parameters))
P = param_dialog.value
pixel_size = P['pixel_size']

figsize = (3, 3) # width, height

df = pd.read_csv(label_path)
df = df[df['mean_z']>=P['min_z']/P['pixel_size']]

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
#f1 = figure(1, figsize=figsize)
#ax1 = f1.subplots()
#prepare_panel(ax1)

sns.jointplot(
    x=mean_z,
    y=z_predict,
    kind="scatter",
    alpha=0.05,
    marker='.',
    s=4,
    marginal_kws=dict(bins=40, fill=True),
    color="blue",
    height=3
)
# # Slide boundaries
xmin, xmax = plt.gca().get_xlim()
plot([xmin, xmax], [xmin, xmax], 'k--')
ylabel(r"z estimate ($\mu$m)")
xlabel(r'z ($\mu$m)')
ylim(xmin-100, xmax)

# ax1.scatter(mean_z, z_predict, alpha=0.01, s=4)
# ax1.plot([zmin, zmax], [zmin, zmax], 'k--')
# ax1.plot(bin_centers, bin_means, "r")
# ax1.set_ylabel(r"z estimate ($\mu$m)")
#
# ax1.set_xlabel(r'z ($\mu$m)')

tight_layout()

savefig(results_path, dpi=300)

show()
