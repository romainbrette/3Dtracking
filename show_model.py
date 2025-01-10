'''
Display model information
'''
import os
from tkinter import filedialog
from tensorflow.keras.models import load_model

## Model
model_filename = filedialog.askdirectory(initialdir=os.path.expanduser('~/Downloads/'), message='Choose a model')

model = load_model(model_filename)

model.summary()
