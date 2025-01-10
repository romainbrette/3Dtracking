# Three-dimensional tracking of protists from two-dimensional microscopy images

## Requirements
- `tensorflow`
- `keras`
- `imageio` (for reading images and movies)

## 2D tracking
This code assumes that 2D tracks have already been extracted from the movie. This can be done with various tools,
such as Fasttrack or Fiji.
The trajectory file can be a Fasttrack file, or a simple csv/tsv file with fields x, y and frame.
Dimensions must be in pixel.

## Typical workflow
1. Make a training data set. This can be done with:
  - `dataset/create_dataset.py` for a movie on a tilted slide;
  - `dataset/create_stack_dataset.py` for movies on horizontal slides at several known z positions;
  - `synthetic/synthetic_dataset.py` for an artificial dataset with blurred ellipses.
  - `synthetic/semi_synthetic_dataset.py` for a semi-artificial dataset made of blurred cell images.

2. Train with `train.py`. Training can be with respect to z or, in the case of a synthetic dataset, sigma
(the amount of blurring).

3. Make predictions on a movie (`predict_movie.py`) or on a dataset (`predict_dataset.py`).

4. Evaluate predictions on a dataset with `evaluation/evaluate_tilted_slide.py` (tilted dataset),
`evaluation/evaluate_stack` (stack dataset) or `evaluate_synthetic` (synthetic dataset).

## File structure

- `train.py` trains the model on a labeled dataset.
- `predict_dataset.py` applies the trained model on a dataset and adds a column `z_predict`.
- `predict_movie.py` applies the trained model on a tiff folder (movie) with trajectory file,
   and adds a column `z` to the trajectory file.

### `dataset/`
Scripts to create labeled datasets of cell images from a tiff folder (movie)
and a trajectory file.

### `synthetic/`
Various scripts to create and analyze synthetic data.

### `gui/`
These are utility scripts for user interfaces.

### `tracking/`
These are utility scripts to load trajectory files.
