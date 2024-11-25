'''
Loads trajectory files
'''
import numpy as np
import pandas as pd
import os

__all__ = ['fasttrack_to_table', 'load_fasttrack', 'magic_load_trajectories']

def fasttrack_to_table(table):
    # Transforms a fasttrack table into a table with standard readable variables
    return pd.DataFrame(data={'x': table['xBody'],
                               'y': table['yBody'],
                               'angle': np.pi - table['tBody'],  # the angle is inverted (head/tail swap)
                               'length': table['bodyMajorAxisLength'],
                               'width': table['bodyMinorAxisLength'],
                               'eccentricity': table['bodyExcentricity'],
                               'frame': table['imageNumber'],
                               'id': table['id']})

def load_fasttrack(filename):
    return fasttrack_to_table(pd.read_csv(filename, sep='\t'))

def magic_load_trajectories(filename):
    '''
    Magically loads a trajectory file by identifying its format.
    '''
    _, ext = os.path.splitext(filename)
    if (ext == '.tsv'): # table with header
        data = pd.read_table(filename)
        # Older files have wrong names
        data = data.rename(
            columns={'centroid-0': 'y', 'centroid-1': 'x', 'major_axis_length': 'length', 'minor_axis_length': 'width',
                     'orientation': 'angle'})
        # Check if there are ids
    elif (ext == '.csv'):
        data = pd.read_csv(filename)
    elif os.path.basename(filename) == 'tracking.txt': # Fasttrack
        data = load_fasttrack(filename)
    else:
        raise OSError("Unknown format")

    return data
