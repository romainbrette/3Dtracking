'''
Extraction of cell images
'''
import pandas as pd
import numpy as np
from scipy import stats

__all__ = ['extract_cells']

def extract_cells(image, data_frame, size, fill_value=0, crop=False, background=None, pixel_size=1.):
    '''
    Extract cell images from a big image with cell coordinates in pixel.
    * `data_frame`: pandas data frame with cell positions in pixel.
    * `size`: size of the output image
    * `background`: if not None, returns the mean background too
    * `crop`: if True, crop a bounding box around the cell based on ellipse parameters
    '''
    image_height, image_width = image.shape

    ## Fill value
    if fill_value is None:
        fill_value, _ = stats.mode(image.flatten())

    ## Iterate over cells
    snippets = []
    for _, row in data_frame.iterrows():
        x0, y0, length, width, angle = (row['x']/pixel_size, row['y']/pixel_size,
                                        row['length']/pixel_size, row['width']/pixel_size, row['angle'])
        snippet = np.ones((size, size), dtype=image.dtype)*fill_value

        if crop:
            # Calculate bounding box from ellipse parameters (works but not very precise):
            bb_width = (length ** 2 * np.cos(angle) ** 2 + width ** 2 * np.sin(angle) ** 2) ** .5
            bb_height = (length ** 2 * np.sin(angle) ** 2 + width ** 2 * np.cos(angle) ** 2) ** .5

            # Scale up to make sure we get the whole cell, and clip to the maximum size
            bb_width = np.clip(bb_width*2, 0, size)
            bb_height = np.clip(bb_height*2, 0, size)
        else:
            bb_width = size
            bb_height = size

        # Shift
        dx = (size-bb_width)//2
        dy = (size-bb_height)//2

        # Crop
        x1, y1 = x0-bb_width//2, y0-bb_height//2
        x2, y2 = np.clip(x1+bb_width, 0, image_width), np.clip(y1+bb_height, 0, image_height)
        dx += -np.clip(x1, -np.inf, 0)
        dy += -np.clip(y1, -np.inf, 0)
        x1 = np.clip(x1, 0, np.inf)
        y1 = np.clip(y1, 0, np.inf)
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        dx, dy = int(dx), int(dy)

        snippet[dy:dy+y2-y1, dx:dx+x2-x1] = image[y1:y2, x1:x2]
        if background is not None:
            background_mean = np.mean(background[y1:y2, x1:x2])
            snippets.append((snippet, background_mean))
        else:
            snippets.append(snippet)

    return snippets