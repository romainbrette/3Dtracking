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

def segments_from_table(table):
    # Extracts a list of uninterrupted segments from a table
    # Copies are returned(not views).
    segments = []
    for _, traj in table.groupby('id', sort=True):
        breaks = list(1+(np.diff(traj['frame'] ) > 1).nonzero()[0]) + [len(traj)]
        n = 0
        for i in breaks:
            segment = pd.DataFrame(traj.iloc[n:i])
            if len(segment) > 2:  # otherwise we can't do any calculation
                segments.append(segment)
            n = i
    return segments

def calculate_speed(segment):
    '''
    Calculate speed in 2D and 3D for a contiguous segment.
    '''
    if len(segment)>2:
        vx, vy = np.diff(segment['x']), np.diff(segment['y'])
        segment['vx'] = np.hstack([vx, nan])
        segment['vy'] = np.hstack([vy, nan])
        speed_2D = (vx**2 + vy**2)**.5
        segment['speed_2D'] = np.hstack([speed_2D, nan])
        if 'z' in segment:
            vz = np.diff(segment['z'])
            segment['vz'] = np.hstack([vz, nan])
            speed_3D = (vx ** 2 + vy ** 2 + vz**2) ** .5
            segment['speed_3D'] = np.hstack([speed_3D, nan])

    return segment

def abs_variation(table):
    '''
    Calculate mean absolute variation of z.
    '''
    vz = table['vz'].values
    vz = vz[~np.isnan(vz)]
    return np.mean(np.abs(vz))
