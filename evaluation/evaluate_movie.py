'''
Evaluate the quality of a model's prediction on trajectories from a movie.
It calculates the mean absolute variation of z within trajectories.
'''
import numpy as np
import os
import pandas as pd
from tkinter import filedialog
from gui.gui import *
import tkinter as tk
from tracking.load_tracking import *
from tracking.trajectory_analysis import *
import yaml

root = tk.Tk()
root.withdraw()

### Files and folders
traj_filename = filedialog.askopenfilename(initialdir=os.path.expanduser('~/Downloads/'), message='Choose a trajectory file')
name, ext = os.path.splitext(traj_filename)
output_file = name+'_eval.yaml'

### Load trajectories
data = magic_load_trajectories(traj_filename)

### Calculate absolute variation of vertical speed
segments = segments_from_table(data)
print(len(segments), 'segments')
processed_table = pd.concat([calculate_speed(segment) for segment in segments])
print(len(processed_table), 'points')
BV = abs_variation(processed_table)

print("Mean absolute variation of z (pix):", BV)
with open(output_file, 'w') as f:
    yaml.dump({'mean_dz': float(BV)}, f)
