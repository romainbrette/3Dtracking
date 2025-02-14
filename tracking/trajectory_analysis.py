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
import warnings
from tqdm import tqdm
try:
    import norfair
except ImportError:
    warnings.warn("Norfair is not available")

def norfair_track(trajectories, distance_threshold=30, memory=10, delay=0, velocity=False, with_z=True, filtered=True):
    '''
    Track objects identified in `trajectories` using Norfair.
    '''
    tracker = norfair.Tracker(distance_function="euclidean", distance_threshold=distance_threshold,
                              initialization_delay=delay, hit_counter_max=memory)

    if velocity: # Kalman based estimate
        trajectories['vx_kalman'] = 0.
        trajectories['vy_kalman'] = 0.
        trajectories['vz_kalman'] = 0.
        trajectories['speed_kalman'] = 0.

    output = []
    nframes = trajectories['frame'].max()+1
    with tqdm(total=nframes, desc="Tracking") as pbar:
        for frame, rows in trajectories.groupby('frame'):
            if with_z:
                norfair_detections = [norfair.Detection(points=np.array([row['x'], row['y'], row['z']]), data=row) for _, row in rows.iterrows()]
            else:
                norfair_detections = [norfair.Detection(points=np.array([row['x'], row['y']]), data=row) for _, row in rows.iterrows()]
            tracked_objects = tracker.update(detections=norfair_detections)
            for object in tracked_objects:
                last_detection = object.last_detection.data
                if last_detection['frame'] == frame: ## the last detection could be far in the past
                    row = last_detection.to_dict()
                    row.update({'id' : object.id})
                    if filtered:
                        if with_z:
                            row['x'], row['y'], row['z'] = object.estimate[0]
                        else:
                            row['x'], row['y'] = object.estimate[0]
                    if velocity:
                        if with_z:
                            vx, vy, vz = object.estimate_velocity[0] # This is quite slow
                            row.update({'vx_kalman' : vx, 'vy_kalman' : vy, 'vz_kalman' : vz,
                                        'speed_kalman' : (vx**2 + vy**2 + vz**2)**.5})
                        else:
                            vx, vy = object.estimate_velocity[0] # This is quite slow
                            row.update({'vx_kalman' : vx, 'vy_kalman' : vy, 'speed_kalman' : (vx**2 + vy**2)**.5})
                    output.append(row)
            pbar.update(1)
    return pd.DataFrame(output) # this line is actually quite slow

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
        segment['vx'] = np.hstack([vx, vx[-1]])
        segment['vy'] = np.hstack([vy, vy[-1]])
        speed_2D = (vx**2 + vy**2)**.5
        segment['speed_2D'] = np.hstack([speed_2D, speed_2D[-1]])
        if 'z' in segment:
            vz = np.diff(segment['z'])
            segment['vz'] = np.hstack([vz, vz[-1]])
            speed_3D = (vx ** 2 + vy ** 2 + vz**2) ** .5
            segment['speed_3D'] = np.hstack([speed_3D, speed_3D[-1]])

    return segment

def abs_variation(table):
    '''
    Calculate mean absolute variation of z.
    '''
    vz = table['vz'].values
    vz = vz[~np.isnan(vz)]
    return np.mean(np.abs(vz))
