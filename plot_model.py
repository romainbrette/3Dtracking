'''
Plot a model
'''
import os
from tkinter import filedialog
from tensorflow.keras.models import load_model
from tensorflow import keras
import tkinter as tk

root = tk.Tk()
root.withdraw()

## Model
model_filename = filedialog.askdirectory(initialdir=os.path.expanduser('~/Downloads/'), message='Choose a model')
img_filename = model_filename+'.png'

model = load_model(model_filename)
keras.utils.plot_model(model, to_file=img_filename, show_shapes=True, show_layer_names=True, rankdir='LR')
