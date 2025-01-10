'''
Evaluate the quality of a model's prediction on trajectories from a movie.
'''
import numpy as np
import os
import pandas as pd
from tkinter import filedialog
from gui.gui import *
import tkinter as tk
from tracking.load_tracking import *
from tracking.trajectory_analysis import *

root = tk.Tk()
root.withdraw()

### Files and folders
traj_filename = filedialog.askopenfilename(initialdir=os.path.expanduser('~/Downloads/'), message='Choose a trajectory file')

### Load trajectories
data = magic_load_trajectories(traj_filename)

### Calculate bounded variation of vertical speed
segments = segments_from_table(data)
print(len(segments), 'segments')
processed_table = pd.concat([calculate_speed(segment) for segment in segments])
print(len(processed_table), 'points')
BV = bounded_variation(processed_table)

print("Bounded variation of z (pix):", BV)
