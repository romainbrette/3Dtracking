'''
From a dataset, makes a big image with cell images vs. z.
'''
import os
from tkinter import filedialog
import tkinter as tk
from gui.gui import *
import pandas as pd
import numpy as np
import tqdm
import yaml
import imageio
import random

n_images = 20

root = tk.Tk()
root.withdraw()

### Folders
path = filedialog.askdirectory(initialdir=os.path.expanduser('~/Downloads/'), message='Choose a dataset folder')
img_path = os.path.join(path, 'images')
label_path = os.path.join(path, 'labels.csv')
big_img_path = os.path.join(path, 'images.png')

## Load data
df = pd.read_csv(label_path)

# Extract filenames and labels
#filenames = df['filename'].values
#z = df['mean_z'].values

m, M = df['mean_z'].min(), df['mean_z'].max()
z_step = (M-m)/n_images

columns = []
for i in range(n_images):
    z_slice = df[(df['mean_z']>=m+i*z_step) & (df['mean_z']<m+(i+1)*z_step)]
    try:
        filenames = random.sample(list(z_slice['filename'].values), n_images)
    except ValueError:
        continue
    images = [imageio.imread(os.path.join(img_path, file)) for file in filenames]
    columns.append(np.vstack(images))

imageio.imwrite(big_img_path, np.hstack(columns))
