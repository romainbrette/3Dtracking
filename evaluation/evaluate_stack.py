'''
Evaluate a trained model on a stacked dataset.

Instead of dealing with pixel size, I just create the dataset with pixel labels rather than um
'''
import os
from tkinter import filedialog
import tkinter as tk
from gui.gui import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
matplotlib.use('TkAgg')
from pylab import *
import yaml
from scipy.stats import linregress

pixel_size = 1.78

root = tk.Tk()
root.withdraw()

### Folders
label_path = filedialog.askopenfilename(initialdir=os.path.expanduser('~/Downloads/'), message='Choose a label file')
results_path = os.path.splitext(label_path)[0]+'_results.png'
info_path = os.path.splitext(label_path)[0]+'_results.yaml'
info = {}

df = pd.read_csv(label_path)
df = df[df['mean_z']>-550] # the recording at -600 is not good
df = df.sort_values(by='mean_z')
df['z_predict'] = df['z_predict']*pixel_size

z_predict, mean_z = df['z_predict'].values, df['mean_z'].values
MAE = np.mean(np.abs(z_predict-mean_z))
print("Mean absolute error:", MAE)
info['MAE'] = float(MAE)

## Linear correlation
result = linregress(z_predict, mean_z)
slope = result.slope
info['slope'] = float(slope)
print("Regression slope:", slope)

fig, (ax1, ax2) = subplots(2, 1, sharex=True)
sns.histplot(df['z_predict'], kde=True, ax=ax1)
ax1.set_xlabel('Estimated z (um)')
sns.histplot(df['mean_z'], kde=True, ax=ax2)
ax2.set_xlabel('z (um)')
suptitle('Distribution of z (um)')
tight_layout()

figure()
title('Estimation error')
df['error'] = (df['z_predict']-df['mean_z']).abs()
sns.violinplot(x='mean_z', y='error', data=df)
xlabel('z (um)')
ylabel('Error (um)')

figure()
suptitle('Mean prediction')
subplot(211)
mean_estimate = df.groupby('mean_z')['z_predict'].mean()
mean_estimate.plot(kind='line', marker='o')
m, M = df['mean_z'].min(), df['mean_z'].max()
plot([m,M], [m,M], 'k--')
xlabel('Mean z (um)')
ylabel('Estimated z (um)')
subplot(212)
std_estimate = df.groupby('mean_z')['z_predict'].std()
precision = float(std_estimate.mean())
info['precision'] = precision
print("Precision:", precision)
std_estimate.plot(kind='line', marker='o')
xlabel('Mean z (um)')
ylabel('Precision (um)')
tight_layout()

savefig(results_path)
with open(info_path, 'w') as f:
    yaml.dump(info, f)

show()
