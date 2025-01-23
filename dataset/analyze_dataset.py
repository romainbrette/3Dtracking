'''
Analyzes a dataset
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
from scipy.stats import linregress
import imageio
import tqdm
import zipfile
import io

root = tk.Tk()
root.withdraw()

### Folders
path = filedialog.askdirectory(initialdir=os.path.expanduser('~/Downloads/'), message='Choose a dataset folder')
img_path = os.path.join(path, 'images')
if os.path.exists(img_path+'.zip'):
    img_path = img_path+'.zip'
    zipped = True
else:
    zipped = False
label_path = os.path.join(path, 'labels.csv')
dataset_parameter_path = os.path.join(path, 'labels.yaml')

## Read data
df = pd.read_csv(label_path)
#df = df.iloc[:500]

## Read dataset parameters
with open(dataset_parameter_path, 'r') as f:
    P_dataset = yaml.safe_load(f)

try:
    z = df['mean_z'].values
except KeyError:
    z = df['z'].values

filenames = df['filename'].values
filenames = [os.path.join(img_path, name) for name in filenames]
n = len(filenames)

## Load images
if zipped:
    with zipfile.ZipFile(img_path, 'r') as zip_ref:
        file_data = [zip_ref.read(name) for name in zip_ref.namelist()]
        images = [imageio.v3.imread(io.BytesIO(zip_ref.read(data)))
                  for data in tqdm.tqdm(zip_ref.namelist(), desc='loading dataset')]
else:
    images = [imageio.v3.imread(os.path.join(img_path, name)) for name in tqdm.tqdm(filenames, desc='loading')]
intensity = np.array([np.mean(image) for image in images])
std = np.array([np.std(image) for image in images])

## Analysis
zmin, zmax = z.min(), z.max()

# Define the number of bins and bin edges
num_bins = 100
bin_edges = np.linspace(zmin, zmax, num_bins + 1)

## Error vs. mean z
# Digitize x_vals to find the bin each x belongs to
bin_indices = np.digitize(z, bin_edges)

# Calculate the mean y value in each bin
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2  # Midpoint of each bin for plotting
bin_means = [np.mean(intensity[bin_indices == i]) for i in range(1, num_bins + 1)]
bin_stds = [np.mean(std[bin_indices == i]) for i in range(1, num_bins + 1)]
bin_contrast = [np.nanmean(intensity[bin_indices == i]/std[bin_indices == i]) for i in range(1, num_bins + 1)]

figure()
subplot(311)
plot(bin_centers, bin_means)
ylabel('Mean')
subplot(312)
plot(bin_centers, bin_stds)
ylabel('Std')
subplot(313)
plot(bin_centers, bin_contrast)
ylabel('Contrast')
xlabel('z')
show()
