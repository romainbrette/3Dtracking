'''
Analysis of trajectories.

Trajectories have the following variables:
    x
    y
    angle
    length
    width
    eccentricity
    frame
    id
'''
import pandas as pd
import numpy as np
from numpy import nan

def filter_shape(table, length=(10., 220.), width=(8., 120.)):
    '''
    Filters based on length and width.
    `length` and `width` are tuples.
    Default values assume um.
    '''
    length_min, length_max = length
    width_min, width_max = width

    return table[(table['length']>=length_min) & (table['length']<=length_max) & (table['width']>=width_min) & (table['width']<=width_max)]

def trajectories_from_table(big_table):
    # Transforms a big dataset into a list of tables, one per trajectory
    # Copies are returned(not views).
    return [pd.DataFrame(traj) for id, traj in big_table.groupby('id', sort=True)]
