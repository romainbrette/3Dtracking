'''
Calculates boundary effects with mean_z from data
'''
import numpy as np
import seaborn as sns
import pandas as pd
from tkinter import filedialog
import tkinter as tk
import matplotlib
matplotlib.use('TkAgg')
from pylab import *
import os

root = tk.Tk()
root.withdraw()

### Folders
path = filedialog.askdirectory(initialdir=os.path.expanduser('~/Downloads/'), message='Choose a dataset folder')
label_path = os.path.join(path, 'labels.csv')

## Read data
df = pd.read_csv(label_path)

## Dataset
noise_amplitude = 260. # 260 - cilia; however, shouldn't I consider a 1.33 factor? I think actually that would tend to reduce noise
mean_z = df['mean_z'].values
m, M = mean_z.min(), mean_z.max()
#z_mean = np.random.uniform(m, M, len(z_mean))
z = mean_z + noise_amplitude*(np.random.rand(len(mean_z))-.5)

df = pd.DataFrame({'z':z, 'mean_z':mean_z})
sampled_df = df.sample(n=5000)

## Calculate E[mean_z | z]
sns.lmplot(data=sampled_df, x='z', y='mean_z', lowess=True, scatter_kws={'alpha': 0.1}, line_kws={'color': 'red', 'linewidth': 2})

show()
