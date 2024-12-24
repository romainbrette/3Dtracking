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
print("Mean absolute error:", np.mean(np.abs(z_predict-mean_z)))

figure()
subplot(211)
sns.histplot(df['z_predict'], kde=True)
xlabel('Estimated z (um)')
subplot(212)
sns.histplot(df['mean_z'], kde=True)
xlabel('z (um)')
suptitle('Distribution of z (um)')
tight_layout()

figure()
title('Estimation error')
df['error'] = (df['z_predict']-df['mean_z']).abs()
sns.violinplot(x='mean_z', y='error', data=df)
xlabel('z (um)')
ylabel('Error (um)')

figure()
title('Mean prediction')
mean_estimate = df.groupby('mean_z')['z_predict'].mean()
mean_estimate.plot(kind='line', marker='o')
xlabel('Mean z (um)')
ylabel('Estimated z (um)')

show()
