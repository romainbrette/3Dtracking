# Three-dimensional tracking of protists from two-dimensional microscopy images

## Requirements
- tensorflow

## 2D tracking
This code assumes that 2D tracking have already be extracted from the movie. This can be done with various tools,
such as Fasttrack or Fiji.

## File structure

- `create_dataset.py` creates a labeled dataset of cell images from a tiff folder (movie) and a trajectory file.
The trajectory file can be a Fasttrack file, or a simple csv file with fields x, y and frame. The movie is taken
on a tilted slide, so that the z position of the slide surface is taken as the label.
- `train.py` trains the model on a labeled dataset.
- `predict_movie.py` applies the trained model on a tiff folder (movie) with trajectory file, and adds a column
`z_predict` to the trajectory file.

### `synthetic/`
Various scripts to create and analyze synthetic data.

### `gui/`
These are utility scripts for user interfaces.

### `tracking/`
These are utility scripts to load trajectory files.
