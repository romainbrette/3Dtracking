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
img_filename = model_filename+'.pdf'

model = load_model(model_filename)
keras.utils.plot_model(model, to_file=img_filename, show_shapes=False, show_layer_names=False)

# dot_graph = keras.utils.model_to_dot(model, show_shapes=True, rankdir="TB")
# dot_graph.write_pdf(img_filename)
