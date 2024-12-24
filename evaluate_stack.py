'''
Evaluate a trained model on a stacked dataset.

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
import seaborn as sns
matplotlib.use('TkAgg')
from pylab import *

root = tk.Tk()
root.withdraw()

### Folders
label_path = filedialog.askopenfilename(initialdir=os.path.expanduser('~/Downloads/'), message='Choose a label file')

df = pd.read_csv(label_path)
df = df.sort_values(by='mean_z')

z_predict, mean_z = df['z_predict'].values, df['mean_z'].values
print("Error vs. mean_z:", (z_predict-mean_z).abs().mean()**.5)

figure()
sns.violinplot(x='mean_z', y='z_predict', data=df)
xlabel('Mean z (um)')
ylabel('Estimated z (um)')

figure()
df['error'] = (df['z_predict']-df['mean_z']).abs()
sns.violinplot(x='mean_z', y='z_predict', data=df)
xlabel('Mean z (um)')
ylabel('Error (um)')

figure()
mean_estimate = df.groupby('mean_z')['z_predict'].mean()
mean_estimate.plot(kind='line', marker='o')
xlabel('Mean z (um)')
ylabel('Estimated z (um)')

show()
